import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
	# GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
	BertConfig, BertForSequenceClassification, BertTokenizer,
	RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
	AutoConfig, AutoModel, AutoTokenizer,
	# DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
)


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
	'codebert': (AutoConfig, AutoModel, AutoTokenizer),
	'graphcodebert': (AutoConfig, AutoModel, AutoTokenizer),
    # 'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def get_model_size(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	model_size = sum([np.prod(p.size()) for p in model_parameters])
	return "{}M".format(round(model_size / 1e+6))


def get_huggingface_path(model: str) -> str:
	if model == 'codebert':
		huggingface_path = 'microsoft/codebert-base'
	elif model == 'graphcodebert':
		huggingface_path = 'microsoft/graphcodebert-base'
	else:
		raise NotImplementedError()
	
	return huggingface_path


def add_special_token(args, config, tokenizer, model, special_token):
	
	special_tokens_dict = {'additional_special_tokens': [special_token]}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	logger.info('We have added %d tokens to tokenizer' % num_added_toks)
	model.resize_token_embeddings(len(tokenizer))
	
	# # Also Update max length of input sequence to args.code_length + args.path_length
	# config.max_position_embeddings = args.code_length + args.path_length + 2
	# tokenizer.model_max_length = args.code_length + args.path_length
	# tokenizer.init_kwargs['max_len'] = args.code_length + args.path_length
	# logger.info('Updated max_length of tokenizer to %d' % tokenizer.model_max_length)
	
	return config, tokenizer, model


def load_model_tokenizer(args):
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	
	# Load the tokenizer
	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case,
												cache_dir=args.cache_dir if args.cache_dir else None)
	logger.info("Finish loading Tokenizer from %s", args.tokenizer_name)

	# Load pretrained model/tokenizer
	config = config_class.from_pretrained(
		args.config_name if args.config_name else args.model_name_or_path,
		trust_remote_code=True,
		revision="main"
	)
	config.num_labels = args.num_labels
	
	# Load the model
	if args.model_name_or_path:
		# model = model_class.from_pretrained(args.model_name_or_path,
		# 									from_tf=bool('.ckpt' in args.model_name_or_path),
		# 									config=config,
		# 									cache_dir=args.cache_dir if args.cache_dir else None)
		model = model_class.from_pretrained(
			args.model_name_or_path,
			trust_remote_code=True,
			revision="main",
		)
	else:
		model = model_class(config)
	
	logger.info("Finish loading Base model [%s] from %s", get_model_size(model), args.model_name_or_path)
	print("Finish loading Base model [%s] from %s" % (get_model_size(model), args.model_name_or_path))
	return model, tokenizer, config



class RobertaClassificationHead(nn.Module):
	
	def __init__(self, config, num_classes):
		super().__init__()
		self.config = config
		
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.out_proj = nn.Linear(config.hidden_size, num_classes)
		
		self.init_weights()
	
	def init_weights(self):
		# Initialize the weights
		self.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		self.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		self.out_proj.bias.data.zero_()
	
	def forward(self, x, **kwargs):
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x


class CodeBERT(nn.Module):
	def __init__(self, encoder, config, tokenizer, args):
		super(CodeBERT, self).__init__()
		self.encoder = encoder
		self.config = config
		self.tokenizer = tokenizer
		self.classifier = RobertaClassificationHead(config, num_classes=args.num_labels)
		self.args = args
	
	def forward(self, input_ids, attention_mask, labels=None):
		outputs = self.encoder(input_ids, attention_mask=attention_mask)
		outputs = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
		
		# Compute the logits using the ClassificationHead
		logits = self.classifier(outputs)
		
		prob = F.softmax(logits, dim=-1)
		if labels is not None:
			labels = labels.float()
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, labels)
			return loss, prob
		else:
			return prob


class MultiPathCodeBERT(nn.Module):
	def __init__(self, encoder, config, tokenizer, args):
		super(MultiPathCodeBERT, self).__init__()
		self.encoder = encoder
		self.config = config
		self.tokenizer = tokenizer
		self.classifier = RobertaClassificationHead(config, num_classes=1)
		self.num_paths = args.num_paths
	
	def forward(self, input_ids, attention_mask, labels=None):
		
		# input_ids: [batch_size, num_paths, seq_length].
		# Do forward pass for each path and max pool the outputs
		
		# First reshape the input_ids and attention_mask to [batch_size * num_paths, seq_length]
		input_ids = input_ids.view(-1, input_ids.size(-1))
		attention_mask = attention_mask.view(-1, attention_mask.size(-1))
		
		outputs = self.encoder(input_ids, attention_mask=attention_mask)
		outputs = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
		logits = self.classifier(outputs)
		
		# Now reshape the logits to [batch_size, num_paths]
		logits = logits.view(-1, self.num_paths)
		# Take max pool over the paths
		logits = torch.max(logits, dim=-1)[0]
		prob = F.sigmoid(logits)

		if labels is not None:
			labels = labels.float()
			# BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class
			loss_fct = nn.BCEWithLogitsLoss()
			loss = loss_fct(logits, labels)
			return loss, prob
		else:
			return prob
		
	def single_path_forward(self, input_ids, attention_mask, labels=None):
		outputs = self.encoder(input_ids, attention_mask=attention_mask)
		outputs = outputs[0][:, 0, :]
		logits = self.classifier(outputs)
		return logits


class GraphCodeBERT(nn.Module):
	def __init__(self, encoder, config, tokenizer, args):
		super(GraphCodeBERT, self).__init__()
		self.encoder = encoder
		self.config = config
		self.tokenizer = tokenizer
		self.classifier = RobertaClassificationHead(config)
		self.args = args
	
	def forward(self, inputs_ids, position_idx, attention_mask, labels=None):
		# embedding
		nodes_mask = position_idx.eq(0)
		token_mask = position_idx.ge(2)
		inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
		nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attention_mask
		nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
		avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
		inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
		
		outputs = self.encoder.roberta(
			inputs_embeds=inputs_embeddings,
			attention_mask=attention_mask,
			position_ids=position_idx,
			token_type_ids=position_idx.eq(-1).long()
		)[0]
		logits = self.classifier(outputs)
		# shape: [batch_size, num_classes]
		prob = F.softmax(logits, dim=-1)
		if labels is not None:
			labels = labels.float()
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, labels)
			return loss, prob
		else:
			return prob
