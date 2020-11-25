from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import json
import math
import torch
from models import Bert
import collections
from random import shuffle
import os
import numpy as np			


class SummarizationDataset(IterableDataset):
	def __init__(self,inputs_dir,to_shuffle=True,is_test=False, dataset_type='train',unit='edu', \
					use_bert=False,bert_model=None,\
					dataset='cnndm',attnmap_type='none',attnmap_path='/scratch/wenxiao/attnmap/'):
		if isinstance(inputs_dir,list):
			self._input_files = inputs_dir
		else:
			inputs_dir = Path(inputs_dir)
			self._input_files = [path for path in inputs_dir.glob("*.pt")]
		self.shuffle=to_shuffle
		self._input_files = sorted(self._input_files)
		if self.shuffle:
			shuffle(self._input_files)
		self.is_test=is_test
		self.unit = unit
		if use_bert:
			self.bert = bert_model
		self.attnmap_type = attnmap_type
		if self.attnmap_type!='none':
			self.attnmap_path = attnmap_path+'%s_%s/'%(attnmap_type,dataset_type)
		self.dataset=dataset
		# self.cur_filenum = 0
		# self._loaddata()


	def _loaddata(self,idx):
		file = self._input_files[idx]
		self.cur_data = torch.load(file)
		if self.shuffle:
			shuffle(self.cur_data)
		if (idx==len(self._input_files)-1) and self.shuffle:
			shuffle(self._input_files)

		# self.cur_filenum+=1
		# self.cur_filenum = self.cur_filenum%len(self._input_files)
	def preprocessing(self,data):
		out = {}
		out['id'] = data['doc_id'].split('.')[0]
		out['src'] = data['src']
		###cnndm
		if self.dataset=='cnndm':
			out['d_labels'] = data['d_labels'][0]
			out['labels'] = data['labels'][0]
			out['segs'] = data['segs']
			out['clss'] = data['clss']+[len(data['segs'])]
		###nyt
		elif self.dataset=='nyt':
			out['d_labels'] = data['d_labels']
			out['labels'] = data['labels']
			out['clss'] = np.where(np.array(out['src'])==101)[0].tolist()+[len(data['src'])]

		out['d_span'] = data['d_span']


		if self.attnmap_type =='none':
			out['attnmap'] = data['attnmap']
		# out['hi_attnmap'] = torch.load(self.attnmap_path+out['id']+'.out.attnmap')
		elif self.attnmap_type =='dep_attnmask':
			out['attnmap'] = torch.tensor(torch.load(self.attnmap_path+out['id']+'.out.attnmap')['par_attention_mask'])
		elif self.attnmap_type =='attn_map_norm':
			out['attnmap'] = torch.tensor(torch.load(self.attnmap_path+out['id']+'.out.attnmap'))
		# if use educlss
		# out['src'],out['clss'],out['segs'],out['d_span'] = self.edu_seg_input(data['src'],data['d_span'])

		out['disco_to_sent'] = self.map_disco_to_sent(out['d_span'])
		if(self.is_test):
			out['disco_txt'] = data['disco_txt']
			out['sent_txt'] = data['sent_txt']
			out['disco_dep'] = data['disco_dep']
			out['tgt_txt'] = '\n'.join(data['tgt_list_str'])

		return out

	def edu_seg_input(self,src,d_span):

		unit_list = [[101]+src[d_span[i][0]:d_span[i][1]]+[102] for i in range(len(d_span))]
		new_src = []
		new_clss=[]
		new_segment=[]
		new_d_span=[]
		for i, unit in enumerate(unit_list):
			if i%2==0:
				new_segment.extend([0]*len(unit))
			else:
				new_segment.extend([1]*len(unit))
			new_clss.append(len(new_src))
			new_src.extend(unit)
			new_d_span.append((new_clss[-1]+1,len(new_src)-1))
		new_clss.append(len(new_src))

		return new_src,new_clss,new_segment,new_d_span

	def map_disco_to_sent(self,disco_span):
		map_to_sent = [0 for _ in range(len(disco_span))]
		curret_sent = 0
		current_idx = 1
		for idx, disco in enumerate(disco_span):
			if disco[0] == current_idx:
				map_to_sent[idx] = curret_sent
			else:
				curret_sent += 1
				map_to_sent[idx] = curret_sent
			current_idx = disco[1]
		return map_to_sent

	def __iter__(self):
		# for i in range(len(self._input_files)):
		if not self.is_test:
			i=0
			while (True):
				self._loaddata(i)
				while len(self.cur_data) !=0:
					data = self.cur_data.pop()

					if 'src' in data.keys() and data['attnmap'] is not None:
						out = self.preprocessing(data)
						yield out 
				i = (i+1)%(len(self._input_files))

		if self.is_test:
			for i in range(len(self._input_files)):
				self._loaddata(i)
				while len(self.cur_data) !=0:
					data = self.cur_data.pop()

					if 'src' in data.keys() and data['attnmap'] is not None:
						out = self.preprocessing(data)
						yield out 

class Batch(object):
	def _pad(self, data, pad_id,width=-1):
		if (width == -1):
			width = max(len(d) for d in data)
		rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
		return rtn_data

	def _cut(self, data):
		if len(data['src'])>self.max_length:
			# self.max_length = max_length
			tmp_clss = [clss for clss in data['clss'] if clss<self.max_length-1]
			data['clss'] = tmp_clss+[data['clss'][len(tmp_clss)]]

			data['labels'] = data['labels'][:(len(data['clss'])-1)]

			end_idx = np.where(np.array(data['src'])==102)[0][len(data['labels'])-1]

			data['src'] = data['src'][:end_idx+1]
			# data['segs'] = data['segs'][:end_idx+1]
			data['d_span'] = [(d[0],min(d[1],end_idx)) for d in data['d_span'] if d[0]<end_idx]
			data['d_labels'] = data['d_labels'][:len(data['d_span'])]
			# data['segs'] = data['segs'][:self.max_length]

			# data['tree_embed'] = data['tree_embed'][:len(data['d_span'])]

			data['attnmap'] = data['attnmap'][:len(data['d_span']),:len(data['d_span'])].type(torch.float)
			# data['hi_attnmap'] = data['hi_attnmap'][:,:len(data['d_span']),:len(data['d_span'])]
		return data

	def _cut_pad_tree_embed(self,tree_embed_batch,length_limit=768, pad_id=0):
		max_num_data= max(len(d) for d in tree_embed_batch)
		pad_sequence = [pad_id]*length_limit
		for i_data, tree_embed in enumerate(tree_embed_batch):
			# pad tree embedding for each position
			for i, tree_embed_s in enumerate(tree_embed):
				if len(tree_embed_s)>=length_limit:
					tree_embed_batch[i_data][i] = tree_embed_batch[i_data][i][:length_limit]
				elif len(tree_embed_s)<length_limit:
					tree_embed_batch[i_data][i] = tree_embed_batch[i_data][i] + [pad_id] * (length_limit - len(tree_embed_batch[i_data][i]))
			# pad tree embedding for each data
			tree_embed_batch[i_data]+=[pad_sequence]*(max_num_data-len(tree_embed_batch[i_data]))
		return tree_embed_batch

	def _build_edu_span(self,d_span_list):
		max_edu = max([len(d_span) for d_span in d_span_list])
		#batch * num_token * num_edu
		edu_span = torch.zeros(self.batch_size,self.max_length,max_edu)
		for i,d_span in enumerate(d_span_list):
			for j, edu in enumerate(d_span):
				edu_span[i,edu[0]:edu[1],j]=1
		return edu_span

	def _build_sent_span(self,clss_list):
		max_sent = max([len(clss) for clss in clss_list])-1
		#batch * num_token * num_sentence
		sent_span = torch.zeros(self.batch_size,self.max_length,max_sent)
		for i,clss in enumerate(clss_list):
			for j, sent in enumerate(clss[:-1]):
				sent_span[i,clss[j]+1:clss[j+1]-1,j]=1
		return sent_span

	def _build_sent_edu_span(self,disco_to_sent,num_edu):
		disco_to_sent = disco_to_sent[:num_edu]
		num_sent = max(disco_to_sent)+1
		span_mat = torch.zeros(num_sent,num_edu)
		for i_edu,i_sent in enumerate(disco_to_sent):
			span_mat[i_sent,i_edu]=1
		return span_mat




	def __init__(self, batch=None, device=None,  is_test=False, max_length=None, unit='edu'):
		"""Create a Batch from a list of examples."""
		if batch is not None:
			self.batch_size = len(batch)
			self.max_length = max([len(x['src']) for x in batch])
			if max_length and (max_length<self.max_length):
				self.max_length=max_length
				batch = [self._cut(d) for d in batch]
			# batch.sort(key=lambda x: len(x['d_span']), reverse=True)

			batch.sort(key=lambda x: len(x['labels']), reverse=True)
			d_span_list = [x['d_span'] for x in batch]
			clss_list = [x['clss'] for x in batch]
			edu_span = self._build_edu_span(d_span_list)
			sent_span = self._build_sent_span(clss_list)
			pre_src = [x['src'] for x in batch]
			pre_d_labels = [x['d_labels'] for x in batch]
			pre_labels = [x['labels'] for x in batch]
			# pre_segs = [x['segs'] for x in batch]
			pre_clss = [x['clss'][:-1] for x in batch]
			# tree_embed = [x['tree_embed'] for x in batch]
			ids = [x['id'] for x in batch]
			pre_attn_map = [x['attnmap'] for x in batch]
			# pre_hi_attn_map = [x['hi_attnmap'] for x in batch]

			src = torch.tensor(self._pad(pre_src, 0))
			d_labels = torch.tensor(self._pad(pre_d_labels, 0)).type(torch.float)
			labels = torch.tensor(self._pad(pre_labels, 0)).type(torch.float)
			# segs = torch.tensor(self._pad(pre_segs, 0))
			# tree_embed = torch.tensor(self._cut_pad_tree_embed(tree_embed))
			mask = (~(src == 0)).type(torch.float)
			
			#batch*edu_num
			edu_mask = (~(torch.sum(edu_span,dim=1) == 0)).type(torch.float)
			sent_mask = (~(torch.sum(sent_span,dim=1) == 0)).type(torch.float)

			clss = torch.tensor(self._pad(pre_clss, -1))
			mask_cls = ~(clss == -1)
			clss[clss == -1] = 0
			if unit=='sent':
				attn_map = torch.zeros(labels.shape[0],labels.shape[1],labels.shape[1])
			else:
				attn_map = torch.zeros(d_labels.shape[0],d_labels.shape[1],d_labels.shape[1])

			for i, attnmap_single in enumerate(pre_attn_map):
				if unit=='sent':			
					span_mat = self._build_sent_edu_span(batch[i]['disco_to_sent'],attnmap_single.shape[1])
					attnmap_single = torch.matmul(torch.matmul(span_mat,attnmap_single.type(torch.float)),span_mat.transpose(1,0))
				# attnmap_single=F.softmax(attnmap_single,dim=1)
				attnmap_single = F.normalize(attnmap_single,p=1,dim=1)

				attn_map[i,:attnmap_single.shape[0],:attnmap_single.shape[1]]=attnmap_single

			setattr(self, 'src_list', pre_src)
			setattr(self, 'd_span_list', d_span_list)

			if unit=='edu':
				setattr(self, 'unit_labels', d_labels.to(device))
				setattr(self, 'unit_mask', edu_mask.to(device))
				setattr(self, 'attn_map', attn_map.to(device))
				# setattr(self, 'hi_attn_map', hi_attn_map.to(device))
			else:
				setattr(self, 'unit_labels', labels.to(device))

				setattr(self, 'unit_mask', sent_mask.to(device))
				# print(attn_map.shape)
				setattr(self, 'attn_map', attn_map.to(device))

			setattr(self, 'ids', ids)
			# setattr(self, 'tree_embed', tree_embed.to(device))

			if (is_test):
				if unit=='edu':
					src_str = [x['disco_txt'] for x in batch]
					setattr(self, 'unit_txt', src_str)
					disco_dep= [x['disco_dep'] for x in batch]
					setattr(self, 'disco_dep', disco_dep)
					disco_to_sent= [x['disco_to_sent'] for x in batch]
					setattr(self, 'disco_to_sent', disco_to_sent)
				else:
					src_str = [x['sent_txt'] for x in batch]
					setattr(self, 'unit_txt', src_str)
				tgt_str = [x['tgt_txt'] for x in batch]
				setattr(self, 'tgt_txt', tgt_str)

	def __len__(self):
		return self.batch_size

class BatchBERTEncoder(object):
	def _cut_pad_tree_embed(self,tree_embed_batch,length_limit=768, pad_id=0):
		max_num_data= max(len(d) for d in tree_embed_batch)
		pad_sequence = [pad_id]*length_limit
		for i_data, tree_embed in enumerate(tree_embed_batch):
			# pad tree embedding for each position
			for i, tree_embed_s in enumerate(tree_embed):
				if len(tree_embed_s)>=length_limit:
					tree_embed_batch[i_data][i] = tree_embed_batch[i_data][i][:length_limit]
				elif len(tree_embed_s)<length_limit:
					tree_embed_batch[i_data][i] = tree_embed_batch[i_data][i] + [pad_id] * (length_limit - len(tree_embed_batch[i_data][i]))
			# pad tree embedding for each data
			tree_embed_batch[i_data]+=[pad_sequence]*(max_num_data-len(tree_embed_batch[i_data]))
		return tree_embed_batch

	def _pad(self, data, pad_id,width=-1):
		if (width == -1):
			width = max(len(d) for d in data)
		rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
		return rtn_data

	def _cut_unit(self,src, max_length):
		if len(src)>max_length:
			src = src[:max_length]
		return src

	def get_bert_representation_single(self,src, bert_model, device, max_length, unit='edu', batch_size=128, d_span=None):

		if unit == 'sentence':
			src = [102] + src
			position_of_sep = np.where(np.array(src)==102)[0]
			unit_list = [self._cut_unit(src[(position_of_sep[i]+1):(position_of_sep[i+1]+1)],max_length) for i in range(len(position_of_sep)-1)]
		elif unit=='edu':
			unit_list = [self._cut_unit(src[d_span[i][0]:d_span[i][1]],max_length) for i in range(len(d_span))]

		output = []
		num_batch = math.ceil(len(unit_list)/batch_size)
		for i in range(num_batch):
			start = i*batch_size
			end = (i+1) * batch_size
			unit_input = unit_list[start:end]
			segs = [[0]*len(unit) for unit in unit_input]
			unit = torch.tensor(self._pad(unit_input, 0)).to(device)
			segs = torch.tensor(self._pad(segs, 0)).to(device)
			mask = torch.ones(segs.shape).to(device)
			out = bert_model(unit,segs,mask)[:,0]
			output.append(out)
		output = torch.cat(output,0) #length * embedding_size
		return output

	def __init__(self, batch=None, device=None,  is_test=False, max_length = 512,bert_model=None, unit='edu'):
		"""Create a Batch from a list of examples."""
		if batch is not None: 
			self.batch_size = len(batch)
			# batch.sort(key=lambda x: len(x['d_span']), reverse=True)
			batch.sort(key=lambda x: len(x['labels']), reverse=True)
			d_span_list = [x['d_span'] for x in batch]

			pre_src = [x['src'] for x in batch]
			pre_d_labels = [x['d_labels'] for x in batch]
			pre_labels = [x['labels'] for x in batch]
			tree_embed = [x['tree_embed'] for x in batch]
			ids = [x['id'] for x in batch]

			d_labels = torch.tensor(self._pad(pre_d_labels, 0)).type(torch.float)
			labels = torch.tensor(self._pad(pre_labels, 0)).type(torch.float)
			tree_embed = torch.tensor(self._cut_pad_tree_embed(tree_embed))
			# bert_representation = []
			# for i in range(len(pre_src)):
			# 	bert_representation.append(self.get_bert_representation_single(pre_src[i], bert_model,device,max_length, d_span=d_span_list[i],unit=unit))
			# # bert_representation =[self.get_bert_representation_single(src_single, bert_model,device,max_length, d_span=d_span_list,unit=unit) for src_single in pre_src]
			# bert_representation = pad_sequence(bert_representation)

			setattr(self, 'd_labels', d_labels.to(device))
			setattr(self, 'labels', labels.to(device))
			setattr(self, 'bert_embedding', bert_representation.to(device))
			setattr(self, 'ids', ids)
			setattr(self, 'tree_embed', tree_embed.to(device))
			setattr(self, 'src_list', pre_src)
			setattr(self, 'd_span_list', d_span_list)

			if (is_test):
				src_str = [x['disco_txt'] for x in batch]
				setattr(self, 'disco_txt', src_str)
				src_str = [x['sent_txt'] for x in batch]
				setattr(self, 'sent_txt', src_str)
				tgt_str = [x['tgt_txt'] for x in batch]
				setattr(self, 'tgt_txt', tgt_str)
				disco_dep= [x['disco_dep'] for x in batch]
				setattr(self, 'disco_dep', disco_dep)
				disco_to_sent= [x['disco_to_sent'] for x in batch]
				setattr(self, 'disco_to_sent', disco_to_sent)


class SummarizationDataLoader(DataLoader):
	def __init__(self,dataset, batch_size=5,device=-1, max_length = 1000, is_test=False,unit='edu'):
		super(SummarizationDataLoader, self).__init__(
			dataset, batch_size=batch_size,collate_fn =self.collate_fn)
		self.max_length = max_length
		self.is_test=is_test
		self.device=device
		# if self.bert_unit_encoder:
		# 	self.max_length=10000
		self.unit = unit

	def collate_fn(self,batch):
		# if self.bert_unit_encoder:
		# 	return BatchBERTEncoder(batch, self.device, self.is_test, self.max_length)
		# else:
		return Batch(batch,self.device,self.is_test,self.max_length,self.unit)





