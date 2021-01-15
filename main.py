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
from run import *
from timeit import default_timer as timer
import random

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-bert_dir", default='../')
	parser.add_argument("-d_v", type=int, default=64)
	parser.add_argument("-d_k", type=int, default=64)
	parser.add_argument("-d_inner", type=int, default=3072)
	parser.add_argument("-d_mlp", type=int, default=100)

	parser.add_argument("-n_layers", type=int, default=2)
	parser.add_argument("-n_head", type=int, default=8)
	parser.add_argument("-dropout", type=float, default=0.1)

	parser.add_argument("-optim", default='adam', type=str)
	parser.add_argument("-lr", default=2e-3, type=float)
	parser.add_argument("-eps", default=1e-9, type=float)
	# parser.add_argument("-weight_decay", default=1e-5, type=float)
	parser.add_argument("-beta1", default= 0.9, type=float)
	parser.add_argument("-beta2", default=0.999, type=float)
	parser.add_argument("-decay_method", default='noam', type=str)
	parser.add_argument("-warmup_steps", default=2400, type=int)
	# parser.add_argument("-max_grad_norm", default=0, type=float)

	parser.add_argument("-inputs_dir", default='./data/train/', type=str)
	parser.add_argument("-val_inputs_dir", default='./data/val/', type=str)

	# input and output directory.
	# parser.add_argument("-train_inputs_dir", default='/scratch/wenxiao/DiscoBERT_CNNDM_tree_embed/train', type=str)
	parser.add_argument("-unit", default='sent', type=str)
	parser.add_argument("-device", default=1, type=int)
	parser.add_argument("-unit_length_limit", default=3, type=int)
	parser.add_argument("-word_length_limit", default=80, type=int)

	parser.add_argument("-batch_size", default=32, type=int)
	parser.add_argument("-load_pretrain", default=False, type=bool)
	parser.add_argument("-pretrained_step", default=10000, type=int)
	parser.add_argument("-attention_type", default='tree', type=str)
	parser.add_argument("-model_name", default='edu_bertunitencoder_attnmap_norm_attnonly_64', type=str)
	#for low resource exp.
	parser.add_argument("-num_train_data", default=None, type=int)
	parser.add_argument("-random_seed", default=1234, type=int)
	parser.add_argument("-dataset", default='cnndm', type=str)
	parser.add_argument("-alternative_attnmap", default='none', type=str)
	args = parser.parse_args()
	print(args)

	inputs_dir = args.inputs_dir
	val_inputs_dir = args.val_inputs_dir

	model_name = args.model_name
	ref_path = './ref_%s/'%(model_name)
	hyp_path = './hyp_%s/'%(model_name)
	save_path = './trained_models/%s/'%(model_name)

	unit = args.unit
	use_edu=(unit=='edu')
	unit_length_limit = args.unit_length_limit
	word_length_limit = args.word_length_limit
	batch_size = args.batch_size


	load_pretrain = args.load_pretrain
	warmup_steps = args.warmup_steps


	if torch.cuda.is_available():
		device = torch.device("cuda:%d"%(args.device))
	else:
		device = torch.device("cpu")
	torch.cuda.set_device(args.device)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if not os.path.exists(hyp_path):
		os.makedirs(hyp_path)
	if not os.path.exists(ref_path):
		os.makedirs(ref_path)




	print('load training data.')

	attnmap_type = args.alternative_attnmap
	print('attention type: %s'%(attnmap_type))

	if args.num_train_data!=None:
		all_files = sorted(os.listdir(inputs_dir))
		random.seed(args.random_seed)
		idx = [random.randint(0,len(all_files)-1) for i in range(0,args.num_train_data)]
		idx=[0]
		print(idx)
		inputs_dir = [inputs_dir+'/'+all_files[i] for i in idx]
		warmup_steps = 1000*args.num_train_data//(batch_size)*2
		print('Warmup steps: %d'%(warmup_steps))

	train = SummarizationDataset(inputs_dir,is_test=False,dataset_type='train', dataset=args.dataset,attnmap_type=attnmap_type)
	val = SummarizationDataset(val_inputs_dir,to_shuffle=False, is_test=True,dataset_type='val',dataset=args.dataset,attnmap_type=attnmap_type)
	train_dataloader = SummarizationDataLoader(train,is_test=False,device=device,batch_size=batch_size, unit=unit)
	val_dataloader = SummarizationDataLoader(val,is_test=True,device=device,batch_size=batch_size,unit=unit)
	
	pos_weight = get_posweight(inputs_dir).to(device)
	# pos_weight = torch.FloatTensor([1]).to(device)
	# pos_weight = torch.FloatTensor([10.11]).to(device)



	model = DiscoExtSumm(args,load_pretrained_bert=True).to(device)
	# optimizer = build_optim(args, model)
	# optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,weight_decay=1e-5, betas=(0.9, 0.98), eps=1e-09)


	optimizer = Optimizer(model.parameters(), lr = args.lr, betas=(args.beta1, args.beta2), eps=args.eps, 
						warmup_steps=warmup_steps)


	best_r2=0
	best_ce=10000
	loss_list = []

	batch_every_step = 1
	report_steps = 50
	eval_steps = 300
	time_start = timer()
	time_epoch_end_old = time_start
	stop_steps=24000
	if args.num_train_data!=None:
		# report_steps = 5*args.num_train_data
		# eval_steps = 30*args.num_train_data
		# stop_steps = 15*eval_steps
		report_steps = 1000*args.num_train_data//(5*batch_size)
		eval_steps = 1000*args.num_train_data//(batch_size)
		stop_steps = 15*eval_steps

		print('Batch Size: %d'%(batch_size))
		print('Report steps: %d'%(report_steps))
		print('Evaluate steps: %d'%(eval_steps))
		print('Stop steps: %d'%(stop_steps))

	## For continue training
	if load_pretrain:
		MODEL_PATH = save_path+'/best_r2'
		if torch.cuda.is_available():
		    model=model.to(device)
		model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
		optimizer._step= args.pretrained_step
		r2,l=evaluate(model,val_dataloader,pos_weight,device, hyp_path, ref_path, \
								word_length_limit=word_length_limit, unit_length_limit=unit_length_limit, \
								use_mmr=use_mmr, lamb=lamb_for_mmr,\
								use_trigram_block=use_trigram_block,use_edu=use_edu)
		best_r2 = r2
		best_ce = l

	model.train()
	print('Start training.')
	for i,batch in enumerate(train_dataloader):
		l = train_batch(model,optimizer,batch,pos_weight,device)
		torch.cuda.empty_cache()
		# r2,l=evaluate(model,val_dataloader,pos_weight,device, hyp_path, ref_path, word_length_limit=1000, unit_length_limit=3,use_edu=use_edu)
		# cur_loss+=l
		loss_list.append(l.cpu())
		# break
		if (i!=0) & (i%batch_every_step==0):
			optimizer.step()
			optimizer.zero_grad()
			if (optimizer._step%report_steps==0):
				print('Step: %d, Loss: %f, Learning rate: %f'%(optimizer._step,sum(loss_list)/len(loss_list),optimizer.current_lr))

				sys.stdout.flush()
			if (optimizer._step % eval_steps==0):
				r2,l=evaluate(model,val_dataloader,pos_weight,device, hyp_path, ref_path, \
								word_length_limit=word_length_limit, unit_length_limit=unit_length_limit, use_edu=use_edu)
				model.train()
				print('Validation loss: %f'%(l))
				time_epoch_end_new = timer()
				print ('Seconds to execute to %d batches: '%(report_steps) + str(time_epoch_end_new - time_epoch_end_old))
				time_epoch_end_old = time_epoch_end_new
				if r2>best_r2:
					PATH = save_path+'/best_r2'
					best_r2 = r2
					torch.save(model.state_dict(), PATH)
					print('saved as best model - highest r2.')
				if l<=best_ce:
					best_ce = l
					print('lowest ce!')
				sys.stdout.flush()	
		if optimizer._step==stop_steps:
			break

