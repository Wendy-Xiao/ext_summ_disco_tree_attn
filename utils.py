from collections import Counter
from pathlib import Path
from random import random
from nltk import word_tokenize
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
import nltk
from nltk.util import ngrams
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
		
def get_meteor(hyp_pathlist,ref_pathlist,model_type):
	all_ref =[]
	all_hyp = []
	total_num = len(hyp_pathlist)
	for i in range(total_num):
		of = open(ref_pathlist[i],'r')
		c = of.readlines()
		c = [i.strip('\n') for i in c]
		of.close()
		all_ref.append(' '.join(c))

		of = open(hyp_pathlist[i],'r')
		c = of.readlines()
		c = [i.strip('\n') for i in c]
		of.close()
		all_hyp.append(' '.join(c))

	of = open('all_ref_inorder.txt','w')
	of.write('\n'.join(all_ref))
	of.close()


	of = open('all_hyp_inorder.txt','w')
	of.write('\n'.join(all_hyp))
	of.close()

	of = open('meteor_out_%s.txt'%(model_type),'w')
	subprocess.call(['java','-Xmx2G','-jar','meteor-1.5/meteor-1.5.jar','all_hyp_inorder.txt','all_ref_inorder.txt','-norm','-f','system1'],stdout=of)
	of.close()

def compute_disco_rouge_single(hyp, bracket_file,edu_file):
	with open(edu_file,'r') as of:
		target = of.readlines()
	while '\n' in target:
		target.remove('\n')
	try:
		root_node= build_tree(bracket_file)
		weight = get_importance_score(root_node,n=0,s=0)
	except:
		weight = {}
		for i in range(len(target)):
			weight[i]=1.0
	score = {}
	score['r-1'] = disco_rouge(weight,target,hyp,1)
	score['r-2'] = disco_rouge(weight,target,hyp,2)
	return score

def get_disco_rouge(all_summaryfiles,all_ids,edu_dir,bracket_dir):
	overall_scores = {}
	overall_scores['r-1']={}
	overall_scores['r-2']={}
	for summary_file, fid in zip(all_summaryfiles,all_ids):
		with open(summary_file,'r') as of:
			hyp = of.read()
		edu_file = edu_dir+'%s.story.doc.out.edus'%(fid)
		bracket_file = bracket_dir+'%s.story.doc.out.bracket'%(fid)
		scores = compute_disco_rouge_single(hyp, bracket_file,edu_file)
		for k1 in scores.keys():
			for k2 in scores[k1]:
				overall_scores[k1][k2]= overall_scores[k1].get(k2,[])+ [scores[k1][k2]]

	print("DRouge-1 r score: %f, DRouge-2 p score: %f, DRouge-1 f-score: %f "%(\
			sum(overall_scores['r-1']['r'])/len(overall_scores['r-1']['r']),\
			sum(overall_scores['r-1']['p'])/len(overall_scores['r-1']['p']),\
			sum(overall_scores['r-1']['f'])/len(overall_scores['r-1']['f'])))
	print("DRouge-2 r score:%f, DRouge-2 p score: %f, DRouge-2 f-score:%f"%(\
			sum(overall_scores['r-2']['r'])/len(overall_scores['r-2']['r']),\
			sum(overall_scores['r-2']['p'])/len(overall_scores['r-2']['p']),\
			sum(overall_scores['r-2']['f'])/len(overall_scores['r-2']['f'])))

def build_tree(bracket_file):
	with open(bracket_file, 'r') as f:
		 # list of nodes containing [[start_span_idx, end_span_idx], Nuclearity]
		json_data = json.load(f)
	#     print(json_data)
	nodes_stack = []
	for node in json_data:
		node_index_tupel = node[0]
		node_index_str = str(node[0])
		node_nuclearity = node[1]
		# Check if node is a leaf
		if node_index_tupel[0] == node_index_tupel[1]:
			nodes_stack.append(Node(idx = node_index_str, nuclearity = node_nuclearity,start=node_index_tupel[0]-1,end=node_index_tupel[1]))
		# Add children for internal nodes
		else:
			tmp_node = Node(idx = node_index_str, nuclearity = node_nuclearity,start=node_index_tupel[0]-1,end=node_index_tupel[1])
			tmp_node.add_child(nodes_stack.pop(), branch = 1)
			tmp_node.add_child(nodes_stack.pop(), branch = 0)
			nodes_stack.append(tmp_node)
	root_node = Node(idx = 'root', nuclearity = None, start=nodes_stack[0].start_idx,end=nodes_stack[1].end_idx)
	root_node.branch = 0
	root_node.add_child(nodes_stack.pop(), branch = 1)
	root_node.add_child(nodes_stack.pop(), branch = 0)
	return root_node

def get_importance_score(node,n=0,s=0):
	if node.nuclearity=='Satellite':
		s+=1
	else:
		n+=1
	if node.num_children==0:
		result_dict={}
		# result_dict[node.start_idx] = (n/(n+s))*(node.start_idx+2)/(node.start_idx+1)
		result_dict[node.start_idx] = (n/(n+s))
		return result_dict
	else:
		result_dict = {}
		if node.children[0]:
			result_dict.update(get_importance_score(node.children[0],n,s))
		if node.children[1]:
			result_dict.update(get_importance_score(node.children[1],n,s))
		return result_dict


def disco_rouge(weight,ref,hyp,n=1,alpha=0.5):
	''' A new ROUGE score taking the discourse strucutre of reference into consideration. We use a new recall while leaving 
		precision unchanged.
		Precision = # match/# ngrams in the candidate.
		Recall = weighted match/sum of weights from the reference
		F1 = precision * recall / ((1 - alpha) * precision + alpha * recall)
	
		target: list of EDUs in the reference, each EDU is a list of tokens.
		weight: dictionary of weight of EDUs in the reference, the weight is computed based on the discourse tree.
		candidates: list of EDUs/Sentences in the candidate summary, each EDU/Sentence is a list of tokens.
		alpha: a float indicating the importance of recall score when computing the f-1 score.
		
	'''
	# Build a dictionary, with ngrams in the reference as the keys, the corresponding importance scores as the values.
	stemmer = nltk.stem.porter.PorterStemmer()
	REMOVE_CHAR_PATTERN = re.compile('[^A-Za-z0-9]')
	target_ngram_with_weight = {}
	ref_list = [nltk.word_tokenize(REMOVE_CHAR_PATTERN.sub(' ', r.lower()).strip()) for r in ref]
	## Stemming
	ref_list_stem = []
	for w_list in ref_list:
		ref_list_stem.append([stemmer.stem(w) for w in w_list])
	ref_list = ref_list_stem
	target_ngrams = [list(ngrams(ref_list[i],n)) for i in range(len(ref_list))]
	for i,edu in enumerate(target_ngrams):
		for ngram in edu:
			# The importance of each ngram depend not only on where it appears, but also on how many times it appears.
			target_ngram_with_weight[ngram] = target_ngram_with_weight.get(ngram,[])+[weight[i]]

	target_ngram_with_weight = {k:sorted(target_ngram_with_weight[k]) for k in target_ngram_with_weight.keys()}
	ref_value=sum([sum(target_ngram_with_weight[k]) for k in target_ngram_with_weight])
	weighted_overlap=0
	match=0
#     candidates = [c for candidate in candidates for c in candidate]
	plain_summary = ' '.join(nltk.sent_tokenize(hyp))
	plain_summary=REMOVE_CHAR_PATTERN.sub(' ', plain_summary.lower()).strip()
	candidates = nltk.word_tokenize(plain_summary)
	candidates = [stemmer.stem(w) for w in candidates]
#     print(candidates)
#     candidate_ngram = build_ngram(candidates,n)
	candidate_ngram = list(ngrams(candidates,n))

	for ngram in candidate_ngram:
		# Start matching with the ngram with the highest weight.
		if len(target_ngram_with_weight.get(ngram,[]))!=0:
#             print(ngram)
			weighted_overlap+=target_ngram_with_weight[ngram].pop()
			match+=1

	scores = {}
#     precision =  0.0 if len(candidates)==0 else match/sum([len(unit) for unit in candidates])
	precision =  0.0 if len(candidate_ngram)==0 else match/len(candidate_ngram)
	scores['p'] = precision
	recall = 0.0 if ref_value==0 else weighted_overlap/ref_value
	scores['r'] = recall

	fscore= 0.0 if (recall == 0.0 or precision == 0.0) else precision * recall / ((1 - alpha) * precision + alpha * recall)
	scores['f'] = fscore
	return scores

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

