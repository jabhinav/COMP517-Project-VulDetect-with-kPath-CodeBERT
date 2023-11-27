import os
from datetime import datetime

import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.model import load_model_tokenizer
from utils.config import get_config, setup_cuda, set_seed

from utils.model import CodeBERT as ModelOfInterest
from utils.data import DefectDataset as CustomDataset
# from utils.data import DefectDataset4Graphs as CustomDataset
# from utils.model import GraphCodeBERT as ModelOfInterest

import logging
import torch

from tqdm import tqdm
import wandb


def evaluate(logger, args, model, tokenizer, file_path, eval_when_training=False):
	eval_output_dir = args.output_dir
	
	_dataset = CustomDataset(tokenizer, args, file_path=file_path)
	
	if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(eval_output_dir)
	
	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(_dataset)
	eval_dataloader = DataLoader(_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
	
	# multi-gpu evaluate
	if args.n_gpu > 1 and eval_when_training is False:
		model = torch.nn.DataParallel(model)
	
	def eval_fn():
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(_dataset))
		logger.info("  Batch size = %d", args.eval_batch_size)
		
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = []
		labels = []
		
		model.eval()
		for batch in eval_dataloader:
			batch = tuple(t.to(args.device) for t in batch)
			inputs = {'input_ids': batch[0],
					  'attention_mask': batch[1],
					  'labels': batch[2]}
						
			with torch.no_grad():
				lm_loss, prob = model(**inputs)
				eval_loss += lm_loss.detach().cpu().mean().item()
				# Save the argmax
				preds.extend(torch.argmax(prob, dim=1).detach().cpu().numpy().tolist())
				labels.extend(torch.argmax(batch[2], dim=1).detach().cpu().numpy().tolist())
			
			nb_eval_steps += 1
		
		# Compute the eval accuracy for n_classes probabilities
		preds = np.array(preds)
		labels = np.array(labels)
		# Accuracy
		eval_acc = np.mean(labels == preds)
		# Precision
		eval_precision = np.mean(labels[preds == 1] == preds[preds == 1])
		# Recall
		eval_recall = np.mean(preds[labels == 1] == labels[labels == 1])
		# F1
		eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall)
		
		eval_loss = eval_loss / nb_eval_steps
		perplexity = torch.tensor(eval_loss)
		
		result = {
			"eval_loss": float(perplexity),
			"eval_acc": round(eval_acc, 4),
			"eval_precision": round(eval_precision, 4),
			"eval_recall": round(eval_recall, 4),
			"eval_f1": round(eval_f1, 4)
		}
		return result
	
	return eval_fn


def train(args, logger):
	
	if args.wandb_logging:
		wandb.init(project="VulDetect", name="codebert-devign")
		wandb.config.update(args)
	
	if args.output_dir is None:
		args.output_dir = args.log_dir
	
	# # Load the model
	model, tokenizer, config = load_model_tokenizer(args)
	model = ModelOfInterest(model, config, tokenizer, args)
	
	# # Load the dataset
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_dataset = CustomDataset(tokenizer, args, file_path=args.train_data_file)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler,  batch_size=args.train_batch_size)
	
	# # Update the step types in the args
	args.max_steps = args.num_train_epochs * len(train_dataloader) if args.max_steps < 0 else args.max_steps
	args.warmup_steps = args.max_steps * 0.1
	# args.save_steps = len(train_dataloader)
	# args.logging_steps = len(train_dataloader)
	
	# # Load optimizer and scheduler
	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
	)
	
	# # multi-gpu training (should be after apex fp16 initialization)
	model.to(args.device)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)
	
	# # Load optimizer and scheduler checkpoint if exists
	checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
	scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
	optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
	if os.path.exists(scheduler_last):
		scheduler.load_state_dict(torch.load(scheduler_last))
	if os.path.exists(optimizer_last):
		optimizer.load_state_dict(torch.load(optimizer_last))
		
	# # Define the evaluation function
	eval_fn = evaluate(logger, args, model, tokenizer, file_path=args.eval_data_file, eval_when_training=True)
	
	# # Log the parameters
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				args.train_batch_size * args.gradient_accumulation_steps * (
					torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", args.max_steps)
	
	train_loss = 0
	global_step = 0
	best_acc = 0
	for idx in range(args.num_train_epochs):
		tr_num = 0
		for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {idx}", position=0, leave=True, total=len(train_dataloader)):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = {'input_ids': batch[0],
					  'attention_mask': batch[1],
					  'labels': batch[2]}
			
			outputs = model(**inputs)
			loss = outputs[0]
			
			if args.n_gpu > 1:
				loss = loss.mean()  # mean() to average on multi-gpu parallel training
			
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps
			
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			
			# # For logging
			tr_num += 1
			train_loss += loss.item()
			
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()
				global_step += 1
				
			
			if args.wandb_logging:
				wandb.log({"train/loss": loss.item()}, step=global_step)
				wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step=global_step)
				
		# # Evaluate
		if args.do_eval:
			eval_results = eval_fn()
			eval_acc = eval_results['eval_acc']
			eval_loss = eval_results['eval_loss']
			
			if args.wandb_logging:
				wandb.log({
					"eval/acc": eval_acc,
					"eval/loss": eval_loss,
					"eval/precision": eval_results['eval_precision'],
					"eval/recall": eval_results['eval_recall'],
					"eval/f1": eval_results['eval_f1']
				}, step=global_step)
			
			if eval_acc >= best_acc:
				logger.info(f"Curr. Best Accuracy: {best_acc} | Curr. Eval Accuracy: {eval_acc}")
				
				best_acc = eval_acc
	
				# # Save the model checkpoint
				output_dir = os.path.join(args.output_dir, 'checkpoint-best')
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
					
				model_to_save = model.module if hasattr(model, 'module') else model
				torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
				logger.info("Saving best model checkpoint to %s", output_dir)

	# # Save the model checkpoint
	output_dir = os.path.join(args.output_dir, 'checkpoint-last')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		
	model_to_save = model.module if hasattr(model, 'module') else model
	torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
	torch.save(args, os.path.join(output_dir, 'training_args.bin'))
	logger.info("Saving model checkpoint to %s", output_dir)
	
	# # Test
	if args.do_test:
		# # Define the testing function
		test_fn = evaluate(logger, args, model, tokenizer, file_path=args.test_data_file, eval_when_training=True)  # Hack
		test_results = test_fn()
		
		print("Test Accuracy: ", test_results['eval_acc'])
		print("Test Precision: ", test_results['eval_precision'])
		print("Test Recall: ", test_results['eval_recall'])
		print("Test F1: ", test_results['eval_f1'])
		
		logger.info("Test Accuracy: %s", test_results['eval_acc'])
		logger.info("Test Precision: %s", test_results['eval_precision'])
		logger.info("Test Recall: %s", test_results['eval_recall'])
		logger.info("Test F1: %s", test_results['eval_f1'])
	
	
def main():
	
	args = get_config()
	args = setup_cuda(args)
	
	# Set the random seed
	set_seed(args.seed)
	
	# Setup logging
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join('./logging', current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	args.log_dir = log_dir
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	# Train (Will test also after training)
	train(args, logger)
	


if __name__ == "__main__":
	main()

