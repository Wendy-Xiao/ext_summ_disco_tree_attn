from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *
from models import *
from simple_optimizer import Optimizer
import sys
from dataloader import SummarizationDataLoader,SummarizationDataset
import argparse
from transformers import BertModel, BertConfig
from run import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-bert_dir", default='/scratch/wenxiao/pretrained_bert/')
	parser.add_argument("-d_v", type=int, default=64)
	parser.add_argument("-d_k", type=int, default=64)
	parser.add_argument("-d_inner", type=int, default=3072)
	parser.add_argument("-d_mlp", type=int, default=100)

	parser.add_argument("-n_layers", type=int, default=2)
	parser.add_argument("-n_head", type=int, default=1)
	parser.add_argument("-dropout", type=float, default=0.1)

	parser.add_argument("-test_inputs_dir", default='./data/test/', type=str)
	parser.add_argument("-unit", default='edu', type=str)
	parser.add_argument("-device", default=2, type=int)
	parser.add_argument("-unit_length_limit", default=6, type=int)
	parser.add_argument("-word_length_limit", default=80, type=int)
	parser.add_argument("-batch_size", default=32, type=int)
	parser.add_argument("-attention_type", default='fixed_rand', type=str)
	parser.add_argument("-model_name", default='edu_bertunitencoder_fixed_rand_lowresource_1', type=str)
	parser.add_argument("-model_path", default='./trained_models/', type=str)
	# parser.add_argument("-use_tree_attnmap", default=False, type=bool)
	parser.add_argument("-dataset", default='cnndm', type=str)
	parser.add_argument("-alternative_attnmap", default='none', type=str)
	args = parser.parse_args()
	print(args)
	# inputs_dir = '/scratch/wenxiao/DiscoBERT_CNNDM/test'
	ref_path = './ref_%s_%s/'%(args.model_name,args.dataset)
	hyp_path = './hyp_%s_%s/'%(args.model_name,args.dataset)
	# test_inputs_dir = '/scratch/wenxiao/DiscoBERT_CNNDM_tree_embed/test/'
	test_inputs_dir = args.test_inputs_dir


	model_name = args.model_name
	MODEL_PATH = args.model_path+ model_name
	bert_config= './bert_config_uncased_base.json'

	unit = args.unit
	use_edu=(unit=='edu')
	unit_length_limit = args.unit_length_limit
	word_length_limit = args.word_length_limit
	batch_size = args.batch_size
	if not os.path.exists(hyp_path):
		os.makedirs(hyp_path)
	if not os.path.exists(ref_path):
		os.makedirs(ref_path)

	if torch.cuda.is_available():
		device = torch.device("cuda:%d"%(args.device))
		torch.cuda.set_device(device)
	else:
		device = torch.device("cpu")

	config = BertConfig.from_json_file(bert_config)

	print('load model')
	model = DiscoExtSumm(args,load_pretrained_bert=False,bert_config=config).to(device)
	if torch.cuda.is_available():
	    model=model.to(device)
	model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
	model.eval()


	# train = SummarizationDataset(inputs_dir,is_test=False)
	# train_dataloader = SummarizationDataLoader(train,is_test=False,device=1,batch_size=8)
	# pos_weight = get_posweight(inputs_dir).to(device)
	pos_weight = torch.FloatTensor([10.11]).to(device)
	print('load data')
	# attention_folder = 'nyt/attn_map_nuc_norm'
	attnmap_type = args.alternative_attnmap
	# attention_folder = 'attn_map_norm'
	print('attention type: %s'%(attnmap_type))
	
	test = SummarizationDataset(test_inputs_dir,to_shuffle=False, is_test=True,dataset_type='test',\
		dataset=args.dataset,attnmap_type=attnmap_type)
	test_dataloader = SummarizationDataLoader(test,is_test=True,device=device,batch_size=batch_size,unit=unit)	
	print('start evaluate')
	r2,l=evaluate(model,test_dataloader,pos_weight,device, hyp_path, ref_path, \
					word_length_limit=word_length_limit, unit_length_limit=unit_length_limit, \
					model_name=model_name,\
					use_edu=use_edu,\
					saveScores=True)
