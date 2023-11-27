import os

import torch

from train_path_codebert_on_devign import evaluate
from utils.model import load_model_tokenizer, add_special_token, MultiPathCodeBERT as ModelOfInterest
from utils.config import get_config, setup_cuda, set_seed
from datetime import datetime
import logging


def test(args, logger):
	if args.output_dir is None:
		args.output_dir = args.log_dir
	
	# Test the best model
	logger.info("***** Running testing *****")
	
	# # Load the model
	model, tokenizer, config = load_model_tokenizer(args)
	config, tokenizer, model = add_special_token(args, config, tokenizer, model, special_token='<EDGE>')
	model = ModelOfInterest(model, config, tokenizer, args)
	
	# Load model checkpoint
	model_to_load = model.module if hasattr(model, 'module') else model
	model_to_load.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'model.pt')))
	logger.info("Loaded model checkpoint from %s", args.ckpt_dir)
	
	# Put model on cuda
	model.to(args.device)
	
	test_fn = evaluate(logger, args, model, tokenizer, file_path=args.test_data_file, path_file=args.test_path_file,
					   eval_when_training=False)
	test_results = test_fn()
	
	print("Test Accuracy: ",  test_results['eval_acc'])
	print("Test Precision: ", test_results['eval_precision'])
	print("Test Recall: ", test_results['eval_recall'])
	print("Test F1: ", test_results['eval_f1'])
	
	logger.info("Test Accuracy: %s",  test_results['eval_acc'])
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
	test(args, logger)


if __name__ == "__main__":
	# $ python test_file.py --ckpt_dir ./logging/multiPathCodebert-devign-k-4/checkpoint-best/ --use_src_code True --num_paths 4
	main()
	