import json
import logging
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.DFG_parser import DFG_python, DFG_java
from utils.program_parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

dfg_function = {
	'python': DFG_python,
	'java': DFG_java,
	# 'ruby': DFG_ruby,
	# 'go': DFG_go,
	# 'php': DFG_php,
	# 'javascript': DFG_javascript,
	# 'c': DFG_c,
	# 'cpp': DFG_cpp,
}

# load parsers
parsers = {}
for lang in dfg_function:
	LANGUAGE = Language('parser/my-languages.so', lang)
	parser = Parser()
	parser.set_language(LANGUAGE)
	parsers[lang] = [parser, dfg_function[lang]]


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, _lang, debug=False):
	"""
	:param code:
	:param _lang:
	:param debug:
	:return:
		code tokens: List[str] is the list of tokens in the code that act as nodes in the DFG i.e. [tok_x_idx]
		dfg: List[str, int, str, List[str], List[int]] is the list of edges in the DFG
			It is of the form [tok_x, tok_x_idx, 'comesFrom', [tok_y], [tok_y_idx]]
	"""
	_parser, _dfg_function = parsers[_lang]
	# remove comments
	try:
		code = remove_comments_and_docstrings(code, _lang)
	except:
		pass
	
	# obtain dataflow
	if _lang == "php":
		code = "<?php" + code + "?>"
	
	try:
		# # Parse Code: Get token to index mapping
		# Get AST
		tree = _parser.parse(bytes(code, 'utf8'))
		root_node = tree.root_node
		# Get index of all the tokens i.e. [((row_start, col_start), (row_end, col_end))]
		tokens_index = tree_to_token_index(root_node)
		code = code.split('\n')
		# Based on the index, get the tokens
		code_tokens = [index_to_code_token(x, code) for x in tokens_index]
		# Map tokens to indices
		index_to_code = {}
		for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
			index_to_code[index] = (idx, code)
		
		# # Get DFG
		try:
			DFG, _ = _dfg_function(root_node, index_to_code, {})
		except:
			DFG = []
		
		DFG = sorted(DFG, key=lambda x: x[1])
		
		# Get nodes corresponding to tokens x and y such that y -> x
		_indexes = set()
		
		for d in DFG:
			# Add x if its value comes from some y
			if len(d[-1]) != 0:
				_indexes.add(d[1])
			# Add all y that are used to compute x
			for x in d[-1]:
				_indexes.add(x)
		# From DFG, add those elements whose x are in indexes
		new_DFG = []
		for d in DFG:
			if d[1] in _indexes:
				new_DFG.append(d)
		dfg = new_DFG
	except:
		dfg = []
		code_tokens = []
	
	if debug:
		print("----- Original code -----")
		print(code)
		print("----- DFG -----")
		for edge in dfg:
			if len(edge[-2]) != 0:  # To avoid printing edges with no source
				print("{} [#{}] <- {} [#{}]".format(edge[0], edge[1], ', '.join(edge[-2]),
													', #'.join([str(x) for x in edge[-1]])))
	return code_tokens, dfg


class GraphInputFeatures(object):
	"""A single training/test features for a example."""
	
	def __init__(self,
				 input_tokens,
				 input_ids,
				 position_idx,
				 dfg_to_code,
				 dfg_to_dfg,
				 idx,
				 label,
				 ):
		self.input_tokens = input_tokens
		self.input_ids = input_ids
		self.position_idx = position_idx
		self.dfg_to_code = dfg_to_code
		self.dfg_to_dfg = dfg_to_dfg
		# label
		self.idx = str(idx)
		self.label = label


def convert_defect_examples_to_graph_features(sample, tokenizer, args):
	# source
	code = ' '.join(sample['func'].split())
	
	# extract data flow
	code_tokens, dfg = extract_dataflow(code, args.lang)
	
	# # tokenize each word in code_tokens s.t. the first token from first word is not pre-prended with 'Ġ', rest all are
	code_tokens: List[List[str]] = [
		tokenizer.tokenize('@ ' + word)[1:] if node_idx != 0
		else tokenizer.tokenize(word) for node_idx, word in enumerate(code_tokens)
	]
	# # ori2cur_pos = {node_idx: (start, end)} s.t. end-start = len(word) of word at node_idx in terms of tokens
	ori2cur_pos = {-1: (0, 0)}
	for i in range(len(code_tokens)):
		ori2cur_pos[i] = (
			ori2cur_pos[i - 1][1],
			ori2cur_pos[i - 1][1] + len(code_tokens[i])
		)
	# # flatten
	code_tokens: List[str] = [y for x in code_tokens for y in x]
	
	# truncate code tokens to adjust for data flow length and 2 special tokens -> start, sep
	code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)][:args.code_length - 2]
	
	# Get source_ids
	source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
	source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
	
	# Get position_idxs
	position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
	
	# Truncate dfg to adjust for code tokens added so far in the source
	dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
	
	# Add the word from end node of each edge as source_token, 0 as position_idx, unk_token_id as source_id
	source_tokens += [x[0] for x in dfg]
	position_idx += [0 for x in dfg]
	source_ids += [tokenizer.unk_token_id for x in dfg]
	
	# Pad
	padding_length = args.code_length + args.data_flow_length - len(source_ids)
	position_idx += [tokenizer.pad_token_id] * padding_length
	source_ids += [tokenizer.pad_token_id] * padding_length
	
	# reindex
	reverse_index = {}  # stores the index of each node in the dfg as {node_idx: edge_idx}
	for edge_idx, edge in enumerate(dfg):
		reverse_index[edge[1]] = edge_idx
	
	# Update dfg
	for edge_idx, edge in enumerate(dfg):
		dfg[edge_idx] = edge[:-1] + ([reverse_index[i] for i in edge[-1] if i in reverse_index],)
	
	# Prepare the mapping from dfg to code and dfg to dfg to be used to prepare the 2D attention mask
	dfg_to_dfg = [x[-1] for x in dfg]
	dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
	length = len([tokenizer.cls_token])
	dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
	
	# Get the label -> Convert the target into one-hot encoding
	label = [0] * args.num_labels
	label[sample['target']] = 1
	
	return GraphInputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, sample['idx'], label)


class DefectDataset4Graphs(Dataset):
	def __init__(self, tokenizer, args, file_path='train'):
		self.examples = []
		self.args = args
		
		samples = []
		with open(file_path) as f:
			for line in f:
				sample = json.loads(line.strip())
				samples.append(sample)
		
		samples = samples[:100]  # Use this for debugging
		for sample in tqdm(samples, desc="Converting to graph features", total=len(samples)):
			self.examples.append(convert_defect_examples_to_graph_features(sample, tokenizer, args))
		
		# For Bookkeeping
		if 'train' in file_path:
			for idx, example in enumerate(self.examples[:3]):
				logger.info("*** Example ***")
				logger.info("idx: {}".format(idx))
				logger.info("label: {}".format(example.label))
				logger.info("input_tokens_1: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
				logger.info("input_ids_1: {}".format(' '.join(map(str, example.input_ids))))
				logger.info("position_idx_1: {}".format(example.position_idx))
				logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code))))
				logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg))))
	
	def __len__(self):
		return len(self.examples)
	
	def __getitem__(self, item):
		# calculate graph-guided masked function -> 2D attention mask
		attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
								self.args.code_length + self.args.data_flow_length), dtype=bool)
		
		# calculate begin index of node and max length of input
		node_index = sum([i > 1 for i in self.examples[item].position_idx])
		max_length = sum([i != 1 for i in self.examples[item].position_idx])
		
		# sequence can attend to sequence
		attn_mask[:node_index, :node_index] = True
		
		# special tokens attend to all tokens
		for idx, i in enumerate(self.examples[item].input_ids):
			if i in [0, 2]:
				attn_mask[idx, :max_length] = True
		
		# nodes attend to code tokens that are identified from
		for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
			if a < node_index and b < node_index:
				attn_mask[idx + node_index, a:b] = True
				attn_mask[a:b, idx + node_index] = True
		
		# nodes attend to adjacent nodes
		for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
			for a in nodes:
				if a + node_index < len(self.examples[item].position_idx):
					attn_mask[idx + node_index, a + node_index] = True
		
		return (torch.tensor(self.examples[item].input_ids),
				torch.tensor(self.examples[item].position_idx),
				torch.tensor(attn_mask),
				torch.tensor(self.examples[item].label))


class InputFeatures(object):
	"""A single training/test features for a example."""
	
	def __init__(self,
				 input_tokens,
				 input_ids,
				 mask,
				 idx,
				 label,
				 ):
		self.input_tokens = input_tokens
		self.input_ids = input_ids
		self.mask = mask
		self.idx = str(idx)
		self.label = label


def convert_defect_examples_to_features(sample, tokenizer, args):
	code = ' '.join(sample['func'].split())
	
	# Tokenize the code
	code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
	source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
	source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
	
	# Pad the rest
	padding_length = args.code_length - len(source_ids)
	source_ids += [tokenizer.pad_token_id] * padding_length
	mask = [1] * len(source_tokens) + [0] * padding_length
	
	# Get the label -> Convert the target into one-hot encoding
	label = [0] * args.num_labels
	label[sample['target']] = 1
	
	return InputFeatures(source_tokens, source_ids, mask, sample['idx'], label)


class DefectDataset(Dataset):
	def __init__(self, tokenizer, args, file_path=None):
		self.examples = []
		samples = []
		with open(file_path) as f:
			for line in f:
				sample = json.loads(line.strip())
				samples.append(sample)
		
		# samples = samples[:100]  # Use this for debugging
		for sample in tqdm(samples, desc="Converting to features", total=len(samples)):
			self.examples.append(convert_defect_examples_to_features(sample, tokenizer, args))
		
		# For Bookkeeping
		if 'train' in file_path:
			for idx, example in enumerate(self.examples[:3]):
				logger.info("*** Example ***")
				logger.info("idx: {}".format(idx))
				logger.info("label: {}".format(example.label))
				logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
				# logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
	
	def __len__(self):
		return len(self.examples)
	
	def __getitem__(self, i):
		return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].mask),
				torch.tensor(self.examples[i].label))


class PathInputFeatures(object):
	"""A single training/test features for a example."""
	
	def __init__(self,
				 input_tokens_seqs,
				 input_ids_seqs,
				 mask_seqs,
				 idx,
				 label,
				 ):
		self.input_tokens_seqs = input_tokens_seqs
		self.input_ids_seqs = input_ids_seqs
		self.mask_seqs = mask_seqs
		self.idx = str(idx)
		self.label = label


def convert_defect_examples_to_path_features(sample, paths, tokenizer, args, special_token='<EDGE>'):
	code = ' '.join(sample['func'].split())
	
	# Tokenize the code
	code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
	
	# Get the paths
	all_paths = []
	# Path are in increasing order of length. Shortest paths are all equal first, k_paths are in increasing order of len
	all_paths.extend(paths['shortest_paths'])
	all_paths.extend(paths['k_simple_paths'])
	
	# Get the source tokens
	source_token_seqs = []

	if len(all_paths) <= args.num_paths:
		selected_paths = all_paths
	else:
		if args.do_train:
			# # For random sampling
			# selected_paths = random.sample(all_paths, args.num_paths)
			# # For deterministic sampling
			selected_paths = all_paths[:args.num_paths]
			# # For sampling from the end
			# selected_paths = all_paths[-args.num_paths:]
		else:
			# For deterministic sampling
			selected_paths = all_paths[:args.num_paths]
	
	assert all([isinstance(path, list) for path in selected_paths])
	
	for path in selected_paths:
		
		if len(path) == 1:
			# Skip the path if it is just a single node
			continue
		
		path_code: str = special_token.join(path)
		path_code: str = ' '.join(path_code.split())
		
		# Tokenize the path
		path_tokens = tokenizer.tokenize(path_code)
		
		# Add code tokens + path tokens
		if args.use_src_code:
			source_token_seqs.append([tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + path_tokens)
		else:
			# [Alternate] Add path tokens only
			source_token_seqs.append([tokenizer.cls_token] + path_tokens + [tokenizer.sep_token])
		
	# Add empty paths
	num_paths_remaining = args.num_paths - len(source_token_seqs)
	for i in range(num_paths_remaining):
		source_token_seqs.append([tokenizer.cls_token] + code_tokens + [tokenizer.sep_token])

	source_id_seqs = []
	mask_seqs = []
	for source_tokens in source_token_seqs:
		source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
		source_ids = source_ids[:args.code_length + args.path_length]
		
		mask = [1] * len(source_ids)
		
		# Pad the rest
		padding_length = args.code_length + args.path_length - len(source_ids)
		source_ids += [tokenizer.pad_token_id] * padding_length
		mask += [0] * padding_length
		
		source_id_seqs.append(source_ids)
		mask_seqs.append(mask)
	
	return PathInputFeatures(source_token_seqs, source_id_seqs, mask_seqs, sample['idx'], sample['target'])


class DefectDataset4Paths(Dataset):
	def __init__(self, tokenizer, args, file_path=None, path_file=None):
		self.examples = []
		samples = []
		with open(file_path) as f:
			for line in f:
				sample = json.loads(line.strip())
				samples.append(sample)
		
		# samples = samples[:100]  # Use this for debugging
		
		with open(path_file) as f:
			path_data = json.load(f)
		
		# samples = samples[:100]  # Use this for debugging
		for sample in tqdm(samples, desc="Converting to features", total=len(samples)):
			# Get the paths
			paths = path_data[str(sample['idx'])]
			self.examples.append(convert_defect_examples_to_path_features(sample, paths, tokenizer, args))
		
		# For Bookkeeping
		if 'train' in file_path:
			for idx, example in enumerate(self.examples[:3]):
				logger.info("*** Example ***")
				logger.info("idx: {}".format(idx))
				logger.info("label: {}".format(example.label))
				for seq in example.input_tokens_seqs:
					logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in seq]))
		# logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
	
	def __len__(self):
		return len(self.examples)
	
	def __getitem__(self, i):
		input_ids_seqs = self.examples[i].input_ids_seqs
		mask_seqs = self.examples[i].mask_seqs
		label = self.examples[i].label
		
		# Stack the sequences
		input_ids_seqs = np.stack(input_ids_seqs, axis=0)
		mask_seqs = np.stack(mask_seqs, axis=0)
		
		return torch.tensor(input_ids_seqs), torch.tensor(mask_seqs), torch.tensor(label)
	
	def get_idxs_for_a_batch(self, start_idx, end_idx):
		return [self.examples[i].idx for i in range(start_idx, end_idx)]
