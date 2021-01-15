from collections import Counter
from pathlib import Path
from random import random
import rouge_papier_v2
import pandas as pd
import re
import numpy as np
import os
import json 
import torch
import os
import subprocess
# import matplotlib.pyplot as plt

# Utility functions
def get_posweight(inputs_dir):
	inputs_dir = Path(inputs_dir)
	all_files = [path for path in inputs_dir.glob("*.pt")]
	total_num=0
	total_pos=0
	for i in range(10):
		data = torch.load(all_files[i])
		for d in data:
			total_num+=len(d['d_labels'][0])
			total_pos+=sum(d['d_labels'][0])
	print('Compute pos weight done! There are %d sentences in total, with %d sentences as positive'%(total_num,total_pos))
	return torch.FloatTensor([(total_num-total_pos)/float(total_pos)])

def make_file_list(input_dir,file_list_file):
	of = open(file_list_file,'r')
	file_list = of.readlines()
	of.close()
	f_list = [Path(input_dir+'/'+f.strip()+'.json') for f in file_list]
	return f_list

def get_all_text(train_input_dir):
	if isinstance(train_input_dir,list):
		file_l = train_input_dir
	else:
		train_input = Path(train_input_dir)
		file_l = [path for path in train_input.glob("*.json")]
	all_tokens = []
	for f in file_l:
		with f.open() as of:
			d = json.load(of)
		tokens = [t for sent in d['inputs'] for t in (sent['tokens']+['<eos>'])]
		all_tokens.append(tokens)
	return all_tokens

def build_word2ind(utt_l, vocabularySize):
	word_counter = Counter([word for utt in utt_l for word in utt])
	print('%d words found!'%(len(word_counter)))
	vocabulary = ["<UNK>"] + [e[0] for e in word_counter.most_common(vocabularySize)]
	word2index = {word:index for index,word in enumerate(vocabulary)}
	global EOS_INDEX
	EOS_INDEX = word2index['<eos>']
	return word2index

# Build embedding matrix by importing the pretrained glove
def getEmbeddingMatrix(gloveDir, word2index, embedding_dim):
	'''Refer to the official baseline model provided by SemEval.'''
	embeddingsIndex = {}
	# Load the embedding vectors from ther GloVe file
	with open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
		for line in f:
			values = line.split()
			word = values[0]
			embeddingVector = np.asarray(values[1:], dtype='float32')
			embeddingsIndex[word] = embeddingVector
	# Minimum word index of any word is 1. 
	embeddingMatrix = np.zeros((len(word2index) , embedding_dim))
	for word, i in word2index.items():
		embeddingVector = embeddingsIndex.get(word)
		if embeddingVector is not None:
			# words not found in embedding index will be all-zeros.
			embeddingMatrix[i] = embeddingVector
	
	return embeddingMatrix


def get_rouge(hyp_pathlist, ref_pathlist, config_path= './config'):
	path_data = []
	uttnames = []
	for i in range(len(hyp_pathlist)):
		path_data.append([hyp_pathlist[i], [ref_pathlist[i]]])
		uttnames.append(os.path.splitext(hyp_pathlist[i])[0].split('/')[-1])

	config_text = rouge_papier_v2.util.make_simple_config_text(path_data)
	config_path = config_path
	of = open(config_path,'w')
	of.write(config_text)
	of.close()
	uttnames.append('Average')
	df,avgfs,conf = rouge_papier_v2.compute_rouge(
		config_path, max_ngram=2, lcs=True, 
		remove_stopwords=False,stemmer=True,set_length = False, return_conf=True)
	df['data_ids'] = pd.Series(np.array(uttnames),index =df.index)
	avg = df.iloc[-1:].to_dict("records")[0]
	c = conf.to_dict("records")
	# if lcs:
	# print(c)
	print("Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f, 95-conf(%f-%f)"%(\
			avg['rouge-1-r'],avg['rouge-1-p'],avg['rouge-1-f'],c[0]['lower_conf_f'],c[0]['upper_conf_f']))
	print("Rouge-2 r score:%f, Rouge-1 p score: %f, Rouge-2 f-score:%f, 95-conf(%f-%f)"%(\
		avg['rouge-2-r'],avg['rouge-2-p'],avg['rouge-2-f'],c[1]['lower_conf_f'],c[1]['upper_conf_f']))
	print("Rouge-L r score:%f, Rouge-1 p score: %f, Rouge-L f-score:%f, 95-conf(%f-%f)"%(\
		avg['rouge-L-r'],avg['rouge-L-p'],avg['rouge-L-f'],c[2]['lower_conf_f'],c[2]['upper_conf_f']))

	return avgfs[1],df
		



if __name__ == '__main__':
	# oracle_path = '/scratch/wenxiao/pubmed/oracle/test/'
	# abstract_path = '/scratch/wenxiao/pubmed/human-abstracts/test/'
	# lead_path = '/scratch/wenxiao/pubmed/lead/test/'
	oracle_path = '/ubc/cs/research/nlp/wenxiao/official_code/test_hyp/oracle-bigpatent_a/'
	lead_path = '/ubc/cs/research/nlp/wenxiao/official_code/test_hyp/lead-bigpatent_a/'
	abstract_path = '/scratch/wenxiao/bigpatent/bigPatentData_splitted/a/human-abstracts/test/'

	d = Path(oracle_path)
	uttnames = [str(path.stem) for path in d.glob("*.txt")]
	lead_pathlist = []
	oracle_pathlist = []
	ref_pathlist = []
	for n in uttnames:
		lead_pathlist.append(lead_path+n+'.txt')
		oracle_pathlist.append(oracle_path+n+'.txt')
		ref_pathlist.append(abstract_path+n+'.txt')

	get_meteor(oracle_pathlist,ref_pathlist,'oracle')
	get_meteor(lead_pathlist,ref_pathlist,'lead')

