from collections import Counter,deque
from random import random
from nltk import word_tokenize
from nltk.util import ngrams
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
import json



def train_batch(model,optimizer,batch,pos_weight,device):

	out,_ = model(batch,device)
	loss = F.binary_cross_entropy_with_logits(out.squeeze(2),batch.unit_labels,weight = batch.unit_mask,reduction='sum',pos_weight=pos_weight)
	# model.zero_grad()
	loss.backward()
	total_num = torch.sum(batch.unit_mask)
	# optimizer.step()
	del batch,out
	return loss.data/total_num

def evaluate(model,dataloader,pos_weight,device, hyp_path, ref_path, \
			word_length_limit=80, unit_length_limit=100,use_edu=True,\
			model_name='bsl',\
			saveScores=False):
	model.eval()
	all_reffiles = []
	all_summaryfiles = []
	all_ids = []
	total_loss = 0
	sigmoid = torch.nn.Sigmoid()
	all_selections = []
	for i,batch in enumerate(dataloader):
		batch_ids, summaryfiles,ref_files,loss,selections = evaluate_batch(model, sigmoid, batch, pos_weight, device, hyp_path,ref_path, \
																word_length_limit, unit_length_limit, use_edu)
		all_summaryfiles.extend(summaryfiles)
		all_reffiles.extend(ref_files)
		all_ids.extend(batch_ids)
		total_loss+=loss
		# print(all_reffiles)
		# print(all_summaryfiles)
		# break
		# all_selections.update(selections)
		all_selections.extend(selections)
		# break
	config_name = hyp_path+'/config'
	rouge2,df = get_rouge(all_summaryfiles, all_reffiles, config_name)
	if compute_drouge:
		get_disco_rouge(all_summaryfiles,all_ids,edu_dir,bracket_dir)
	if saveScores:
		all_selections.append([0])
		all_ids.append('avg')
		# all_attn_weight.append([0])
		df['id'] = pd.Series(all_ids,index =df.index)
		df['selections'] = pd.Series(np.array(all_selections),index =df.index)
		df.to_csv('./csv_output/%s.csv'%(model_name))

	# of = open('./selection_files/selections_test_%s.json'%(model_name),'w')
	# all_selections = json.dump(all_selections,of)
	# of.close()
	return rouge2, total_loss/i


def evaluate_batch(model, sigmoid,batch, pos_weight, device, hyp_path,ref_path,  \
					word_length_limit, unit_length_limit, use_edu=True,\
					use_mmr=False, lamb=0.6,\
					use_trigram_block=False):
	out,unit_repre = model(batch,device)
	# batch * length
	loss = F.binary_cross_entropy_with_logits(out.squeeze(2),batch.unit_labels,weight = batch.unit_mask,reduction='sum',pos_weight=pos_weight)
	out = out.squeeze(-1)
	scores = sigmoid(out).data
	# scores = scores.permute(1,0) # length * batch
	if use_edu:
		summaryfiles, ref_files,selections= predict(scores.cpu(), batch.ids, batch.unit_txt, hyp_path, ref_path,\
									word_length_limit, unit_length_limit, batch.tgt_txt,
									use_edu, batch.disco_dep, batch.disco_to_sent)
	else:
		summaryfiles, ref_files,selections= predict(scores.cpu(), batch.ids, batch.unit_txt, hyp_path, ref_path,\
									word_length_limit, unit_length_limit, batch.tgt_txt,
									use_ed)
	batch_ids = batch.ids
	total_num = torch.sum(batch.unit_mask)
	del batch,out,scores
	return batch_ids, summaryfiles,ref_files,loss.data/total_num,selections

def predict(score_batch, ids, src_txt_list, hyp_path,ref_path,\
			word_length_limit, unit_length_limit, tgt_txt,\
			use_edu, disco_dep=None, disco_to_sent=None):
	#score_batch = [batch,seq_len]
	summaryfile_batch = []
	reffile_batch = []
	# all_ids = []
	selections=[]
	for i in range(len(src_txt_list)):
		summary = []
		scores = score_batch[i,:len(src_txt_list[i])]
		sorted_linenum = [x for _,x in sorted(zip(scores,list(range(len((src_txt_list[i]))))),reverse=True)]
		if use_edu:
			dep_dict = build_dep_dict(disco_dep[i])
		# sorted_linenum = build_candidates(scores,dep_dict)
		selected_ids = [] 
		wc = 0
		uc = 0

		for j in sorted_linenum:
			# use edu
			if use_edu:
				cur_select = select_edus_with_dependent(j,dep_dict)
				# cur_select=j
				cur_select = list(set(cur_select)-set(selected_ids))
				# use trigram trick
				#update selected summary
				for idx in cur_select:
					summary.append(' '.join(src_txt_list[i][idx]))
					# selected_ids.append(j)
					wc+=len(src_txt_list[i][idx])
					uc+=1
				selected_ids.extend(cur_select)
			# use sentence
			else:
				selected_ids.append(j)
				summary.append(' '.join(src_txt_list[i][j]))
				wc+=len(src_txt_list[i][j])
				uc+=1

			if uc>=unit_length_limit:
				break
			if wc>=word_length_limit:
				break
		
		if use_edu:
			order_summ = [(x,y) for x,y in sorted(zip(selected_ids,summary),reverse=False)]
			summary =build_final_summary(order_summ,disco_to_sent[i])
		else:
			order_summ = [y for _,y in sorted(zip(selected_ids,summary),reverse=False)]
			summary='\n'.join(order_summ)
		# summary='\n'.join(inorder_summary)

		selections.append(selected_ids)
		fname = hyp_path+ids[i]+'.hyp.txt'
		of = open(fname,'w')
		of.write(summary)
		of.close()
		summaryfile_batch.append(fname)

		refname = ref_path+ids[i]+'.ref.txt'
		of = open(refname,'w')
		of.write(tgt_txt[i])
		of.close()
		reffile_batch.append(refname)

	return summaryfile_batch, reffile_batch,selections

def build_dep_dict(dep):
	dic = {}
	for d in dep:
		source_node, tgt_node = d
		if source_node == tgt_node:
			continue
		if source_node - 1 in dic:
			dic[source_node - 1] = list(set(dic[source_node - 1] + [tgt_node - 1]))
		else:
			dic[source_node - 1] = [tgt_node - 1]

	return dic

def build_candidates(scores,dep_dict):
	candidates = []
	candidate_scores = []
	for i in range(len(scores)):
		selections = select_edus_with_dependent(i,dep_dict)

		candidate_scores.append(torch.tensor([scores[j] for j in selections]).mean())
		candidates.append(selections)
	# print(candidate_scores)
	candidate_ordered = [x for _,x in sorted(zip(candidate_scores,candidates),reverse=True)]
	# print(candidate_ordered)
	# print(scores)
	return candidate_ordered


def select_edus_with_dependent(idx,dep_dict):
	count = 0
	selected = deque()
	output = []
	selected.append(idx)
	while len(selected)!=0:
		cur_idx = selected.popleft()
		output.append(cur_idx)
		if cur_idx in dep_dict:
			selected.extend(dep_dict[cur_idx])
	return list(set(output))

def build_final_summary(order_summ, disco_to_sent):
	d = {}
	for edu_idx,txt in order_summ:
		sent_num = disco_to_sent[edu_idx]
		if sent_num in d:
			d[sent_num]= d[sent_num]+' '+txt
		else:
			d[sent_num] = txt
	final_summ = []
	for k in sorted(d.keys()):
		final_summ.append(d[k])
	return '\n'.join(final_summ)

# return true if there is no same trigram, false otherwise.
def check_trigram(current_trigrams, units, edu=True):
	if edu:
		new_trigrams = set()
		for unit in units:
			new_trigrams=new_trigrams.union(set(ngrams(unit,3)))
		# print(new_trigrams)
	else:
		new_trigrams = set(ngrams(units,3))
	return len(current_trigrams.intersection(new_trigrams))==0, current_trigrams.union(new_trigrams)


