import os
from nltk import sent_tokenize,word_tokenize
import re
import sys
from pathlib import Path
import torch

def split_by_sentences(in_file,out_file):
	of = open(in_file,'r')
	text=of.read()
	of.close()
	text=re.sub('""','',text)
	text=re.sub("''"'','',text)
	text=re.sub('``','',text)
	text=re.sub(' ` ','',text)
	text=re.sub(" ' ",'',text)
	sents = sent_tokenize(text)
	of = open(out_file,'w')
	of.write('\n'.join(sents))
	of.close()

def preprocess_sent(text):
	text=re.sub('""','',text)
	text=re.sub("''"'','',text)
	text=re.sub('``','',text)
	text=re.sub(' ` ','',text)
	text=re.sub(" ' ",'',text)
	return text

def get_sentseg_cnndm(data_folder,sentseg_folder):
	files = [f for f in Path(data_folder).glob('*.pt')]
	num_data = 0
	for i_file,f in enumerate(files):
		all_data = torch.load(f)
		new_data = []
		num_data+=len(all_data)
		for data in all_data:
			doc_id = data['doc_id']
			sentences = [' '.join(s) for s in data['sent_txt']]
		# 	sentences = [preprocess_sent(' '.join(s)) for s in data['sent_txt']]
		# 	if len(sentences)==0:
		# 		print('empty')
		# 		continue
		# 	new_sent_text = [word_tokenize(s) for s in sentences]
		# 	new_sent_text = [s for s in new_sent_text if s!=[]]
		# 	if all(map(str.isupper,new_sent_text[-1])) :
		# 		# print(new_sent_text[-1])
		# 		data['catogry'] = ' '.join(new_sent_text[-1])
		# 		new_sent_text = new_sent_text[:-1]
		# 	data['sent_txt'] = new_sent_text
			# new_data.append(data)
			# print(data)

			of = open(sentseg_folder+doc_id+'.out','w')
			of.write('\n'.join(sentences))
			of.close()
		# torch.save(all_data,f)
		print('%d/%d: %s is done!'%(i_file,len(files),f.name))
		# break
	print(num_data)


def get_sentseg_tac(data_folder,sentseg_folder):
	files = os.listdir(data_folder)
	for i_file,f in enumerate(files):
		with open(data_folder+f,'r',encoding='utf-8',errors='ignore') as of:
			lines = of.readlines()
		sents = []
		for l in lines:
			if l!='\n':
				sents.extend(sent_tokenize(l))
		of = open(sentseg_folder+f+'.out','w')
		of.write('\n'.join(sents))
		of.close()
	print('Done, %d data in total.'%(i_file))


if __name__ == '__main__':
	all_datasets = ['train','valid','test']
	for dataset in all_datasets:
		# in_dir = '/scratch/wenxiao/nyt_dataset/article/%s/'%(dataset)
		in_dir = '/scratch/wenxiao/NYT_data/%s/'%(dataset)
		out_dir = '/scratch/wenxiao/nyt_dataset_discourse/sentseg/%s/'%(dataset)
		new_datadir = ' /scratch/wenxiao/NYT_data_withedu/%s/'%(dataset)
		EDUout_dir = '/scratch/wenxiao/nyt_dataset_discourse/eduseg/%s/'%(dataset)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		if not os.path.exists(EDUout_dir):
			os.makedirs(EDUout_dir)
		# all_files = os.listdir(in_dir)
		# print('%s start segmenting.'%(dataset))
		# for i in range(len(all_files)):
		# 	f = all_files[i]
		# 	in_file = in_dir+f
		# 	out_file = out_dir+f[:-4]+'.out'
		# 	split_by_sentences(in_file,out_file)
		# 	if i%10000==0:
		# 		print('%d data done'%(i))
		# 		sys.stdout.flush()
		# get_sentseg(in_dir,out_dir)
		# os.system('python run.py --segment --input_dir %s --result_dir %s --output_dir %s --gpu 2 --batch_size 16'%(in_dir,EDUout_dir,new_datadir))

		# all_files = os.listdir(EDUout_dir)
		# for i in range(len(all_files)):
		# 	f = all_files[i]
		# 	of = open(EDUout_dir+f,'r')
		# 	all_edus = of.readlines()
		# 	of.close()

		# 	new_edus = []
		# 	for edu in all_edus:
		# 		edu = edu.strip()
		# 		if edu !='':
		# 			new_edus.append(edu)
		# 	of = open(EDUout_dir+f,'w')
		# 	of.write('\n'.join(new_edus))
		# 	of.close()
		# 	if i%10000==0:
		# 		print('%d data done'%(i))
		# 		sys.stdout.flush()
	year = ['2008','2009']
	tp = ['models','peers']
	for y in year:
		for t in tp:
			in_dir = '/scratch/wenxiao/TAC/TAC%s/%s/'%(y,t)
			sentseg_dir = '/scratch/wenxiao/TAC/TAC_discourse/sentseg/TAC%s/%s/'%(y,t)
			eduseg_dir = '/scratch/wenxiao/TAC/TAC_discourse/eduseg/TAC%s/%s/'%(y,t)
			if not os.path.exists(sentseg_dir):
				os.makedirs(sentseg_dir)
			if not os.path.exists(eduseg_dir):
				os.makedirs(eduseg_dir)
			print('Processing %s %s.'%(y,t))
			get_sentseg_tac(in_dir,sentseg_dir)
			os.system('python run.py --segment --input_dir %s --result_dir %s --gpu 1 --batch_size 16'%(sentseg_dir,eduseg_dir))


