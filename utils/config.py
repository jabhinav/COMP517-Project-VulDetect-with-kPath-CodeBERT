import argparse
import multiprocessing
import os
import random

import numpy as np
import torch
import logging

from utils.model import get_huggingface_path

logger = logging.getLogger(__name__)


def get_config() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	
	model_type = "codebert"
	parser.add_argument("--wandb_logging", type=bool, default=True)
	
	# Main params
	parser.add_argument("--num_paths", default=1, type=int, help="Number of paths for classification")
	parser.add_argument("--use_src_code", default=False, type=bool)
	
	# # Specify paths here
	parser.add_argument("--train_data_file", default="./Devign/train.jsonl", type=str,
						help="The input training data file (a text file).")
	parser.add_argument("--eval_data_file", default="./Devign/valid.jsonl", type=str,
						help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
	parser.add_argument("--test_data_file", default="./Devign/test.jsonl", type=str,
						help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
	
	parser.add_argument("--train_path_file", default="./Devign/PDG/train_CDG_REACHING_DEF_path_data.json", type=str)
	parser.add_argument("--eval_path_file", default="./Devign/PDG/valid_CDG_REACHING_DEF_path_data.json", type=str)
	parser.add_argument("--test_path_file", default="./Devign/PDG/test_CDG_REACHING_DEF_path_data.json", type=str)
	
	# parser.add_argument("--train_path_file", default="./Devign/CFG/train_CFG_path_data.json", type=str)
	# parser.add_argument("--eval_path_file", default="./Devign/CFG/valid_CFG_path_data.json", type=str)
	# parser.add_argument("--test_path_file", default="./Devign/CFG/test_CFG_path_data.json", type=str)
	
	parser.add_argument("--output_dir", default=None, type=str,
						help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--ckpt_dir", default=None, type=str,
						help="The output directory where the model predictions and checkpoints will be written.")
	
	# # Specify model type here
	huggingface_path = get_huggingface_path(model_type)
	parser.add_argument("--model_type", default=model_type, type=str, choices=['codebert', 'graphcodebert'],
						help="The model architecture to be fine-tuned.")
	parser.add_argument("--model_name_or_path", default=huggingface_path, type=str,
						help="The model checkpoint for weights initialization.")
	parser.add_argument("--config_name", default=huggingface_path, type=str,
						help="Optional pretrained config name or path if not the same as model_name_or_path")
	parser.add_argument("--tokenizer_name", default=huggingface_path, type=str,
						help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
	parser.add_argument("--num_labels", default=2, type=int, help="Number of labels for classification")
	
	parser.add_argument("--code_length", default=384, type=int)
	parser.add_argument("--data_flow_length", default=128, type=int)
	parser.add_argument("--path_length", default=128, type=int)

	
	parser.add_argument("--do_train",  default=True,
						help="Whether to run training.")
	parser.add_argument("--do_eval", default=True,
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_test", default=True,
						help="Whether to run eval on the dev set.")
	
	parser.add_argument("--cache_dir", default="", type=str,
						help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
	parser.add_argument("--do_lower_case", action='store_true',
						help="Set this flag if you are using an uncased model.")
	
	parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
						help="Batch size per GPU/CPU for training. (default: 8)")
	parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
						help="Batch size per GPU/CPU for evaluation. (default: 16)")
	parser.add_argument("--num_train_epochs", default=5, type=int,
						help="Total number of training epochs to perform. (default: 5)")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--learning_rate", default=2e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
						help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")
	
	parser.add_argument('--logging_steps', type=int, default=50,
						help="[Unused] Log every X updates steps.")
	parser.add_argument('--save_steps', type=int, default=50,
						help="[Unused] Save checkpoint every X updates steps.")
	parser.add_argument("--no_cuda", action='store_true',
						help="Avoid using CUDA when available")
	parser.add_argument('--seed', type=int, default=123456,
						help="random seed for initialization")
	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument('--fp16_opt_level', type=str, default='O1',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training: local_rank")
	
	args = parser.parse_args()
	
	return args


def setup_cuda(args: argparse.Namespace) -> argparse.Namespace:
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1
	
	args.device = device
	
	cpu_cont = multiprocessing.cpu_count()
	logger.info(f"cpu count: {cpu_cont}")
	args.cpu_cont = cpu_cont
	
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
				   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
	
	return args


def set_seed(seed=42):
	random.seed(seed)
	os.environ['PYHTONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
