import os
import pickle
import torch
import nltk
import json
import glob
import re
import numpy as np
import sys

class Node(object):
	def __init__(self, idx, nuclearity,start,end):
		self.idx = idx
		self.nuclearity = nuclearity
		self.parent = None
		self.num_children = 0
		self.children = [None, None]
		self.positional_encoding = None
		self.branch = None
		self.start_idx = int(start)
		self.end_idx = int(end)

	def add_child(self,child, branch):
		child.parent = self
		self.num_children += 1
		child.branch = branch
		self.children[branch] = child

	def set_height(self,height):
		self.height = height

	def set_level(self,level):
		self.level = level


class Dep_Node(object):
	def __init__(self, idx):
		self.idx = idx
#         self.text = text
		self.sentence = None
		self.parent = None
		self.positional_encoding = None
		self.num_children = 0
		self.children = []

	def add_child(self, child):
		child.parent = self
		self.num_children += 1
		self.children.append(child)
		
	def remove_child(self, child):
		self.num_children -= 1
		self.children.remove(child)

def set_height_bottomup(node):
	if not node:
		return -1
	else:
		h0 = set_height_bottomup(node.children[0])
		h1 = set_height_bottomup(node.children[1])
		current_height = max(h0,h1)+1
		node.set_height(current_height)
		return current_height

def set_level_topdown(node,cur_level):
	if not node:
		return -1
	else:
		node.set_level(cur_level)
		l0=set_level_topdown(node.children[0],cur_level+1)
		l1=set_level_topdown(node.children[1],cur_level+1)
		return max(l0,l1,cur_level)
# def get_num_leaves(node):
# 	if not node:
# 		return 0
# 	elif node.num_children==0:
# 		return 1
# 	else:
# 		return get_num_leaves(node.children[0])+get_num_leaves(node.children[1])


def build_attention_map_random(root):
	num_leaves = root.end_idx-root.start_idx
	# att_map = np.zeros((height+1,num_leaves,num_leaves))
	random_attention_map=np.random.rand(num_leaves,num_leaves)
	return random_attention_map

def build_attention_map(root,height):
	num_leaves = root.end_idx-root.start_idx

	att_map = np.zeros((height+1,num_leaves,num_leaves))

	return build_attention_map_bottomup(root,att_map)

def build_attention_map_bottomup(node,att_map):
	if not node:
		return att_map
	h = att_map.shape[0]-1
	if node.nuclearity=='Nucleus':
		#### if without nucleus, set the number as 1 here.
		att_map[h-node.level,node.start_idx:node.end_idx,node.start_idx:node.end_idx]=2
	else:
		att_map[h-node.level,node.start_idx:node.end_idx,node.start_idx:node.end_idx]=1
	attn_map = build_attention_map_bottomup(node.children[0],att_map)
	attn_map = build_attention_map_bottomup(node.children[1],att_map)
	return att_map

def get_importance_score(node,n=0,s=0):
	if node.nuclearity=='Satellite':
		s+=1
	else:
		n+=1
	if node.num_children==0:
		result_dict={}
		result_dict[node.start_idx] = (n/(n+s))
		return result_dict
	else:
		result_dict = {}
		if node.children[0]:
			result_dict.update(get_importance_score(node.children[0],n,s))
		if node.children[1]:
			result_dict.update(get_importance_score(node.children[1],n,s))
		return result_dict

def build_tree(bracket_file):
	with open(bracket_file, 'r') as f:
		 # list of nodes containing [[start_span_idx, end_span_idx], Nuclearity]
		json_data = json.load(f)
		# print(json_data)
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

# Generate dependency tree from constituency tree
def map_disco_to_sent(disco_span):
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

def set_head(node):
	if node.num_children==0:
		node.head=node.start_idx
	else:
		for child in node.children:
			set_head(child)
		if node.children[0].nuclearity!='Satellite':
			node.head = node.children[0].head
		else:
			node.head = node.children[1].head
			
def const_to_dep_tree(node):
	set_head(node)
	const_leaves = get_leaf_nodes(node)
	dep_nodes = [Dep_Node(l.start_idx) for l in const_leaves]
	for leaf,dep_node in zip(const_leaves,dep_nodes):
		closest_S_ancestor, is_tree_head = find_S_ancestor(leaf)
		if is_tree_head:
			root = dep_node
		else:
			dep_node.parent=dep_nodes[closest_S_ancestor.head]
			dep_nodes[closest_S_ancestor.head].add_child(dep_node)
	for dep_node in dep_nodes:
		# print('current id:%d'%(dep_node.idx))
		child_id = []
		for child in dep_node.children:
			child_id.append(child.idx)
		# print(child_id)
	return dep_nodes
		
def find_S_ancestor(const_node):
	if const_node.nuclearity == None and const_node.parent == None:
		return const_node, True
	if const_node.nuclearity == 'Nucleus' and not const_node.parent == None:
		closest_S_ancestor, is_tree_head = find_S_ancestor(const_node.parent)
	else:
		closest_S_ancestor = const_node.parent
		is_tree_head = False
	return closest_S_ancestor, is_tree_head

def make_dep_attention_mask(node_list):
	num_edus = len(node_list)
	par_attention_mask = np.zeros((num_edus,num_edus))
	chil_attention_mask = np.zeros((num_edus,num_edus))
	
	for idx,node in enumerate(node_list):
		if node.parent==None:
			par_attention_mask[idx,idx]=1
		else:
			par_attention_mask[idx,node.parent.idx]=1
		if node.num_children==0:
			chil_attention_mask[idx,idx]=1
		else:
			for child in node.children:
				chil_attention_mask[idx,child.idx]=1/node.num_children
	return par_attention_mask,chil_attention_mask

def get_leaf_nodes(node):
	if node.num_children == 0:
		return [node]
	else:
		leaves = []
		for child_node in node.children:
			leaves.extend(get_leaf_nodes(child_node))
		return leaves

if __name__ == '__main__':
	####CNNDM
	for d in ['train','val','test']:
		# save_trees_folder = '/scratch/discourse_application_for_summarization/cnndm/brackets_%s/'%(d)
		# save_attnmap_folder = '/scratch/wenxiao/attnmap/dep_attnmask_%s/'%(d)
		# save_attnmap_folder = '/scratch/wenxiao/attnmap/rand_attnmask_%s/'%(d)
		save_attnmap_folder = '/scratch/wenxiao/attnmap/attn_map_norm_%s/'%(d)
		#save_encodings_folder = '/scratch/discourse_application_for_summarization/discourse_pos_embeddings_val'
		#save_encodings_folder = '/scratch/discourse_application_for_summarization/discourse_pos_embeddings_test'
		if not os.path.exists(save_attnmap_folder):
			os.mkdir(save_attnmap_folder)

		bracket_files = [os.path.join(save_trees_folder, fname) for fname in os.listdir(save_trees_folder) if fname.endswith('.out.bracket')]
		level_dict = {}
		for idx,bracket_file in enumerate(bracket_files):
			root_node=build_tree(bracket_file)

			###### Build attention map fpr dep tree
			# dep_nodes=const_to_dep_tree(root_node)
			# par_attention_mask,chil_attention_mask=make_dep_attention_mask(dep_nodes)
			# out = {}
			# out['par_attention_mask'] = par_attention_mask
			# out['chil_attention_mask']=chil_attention_mask

			##### Build attention map for constituency tree
			height = set_height_bottomup(root_node)
			max_level=set_level_topdown(root_node,0)
			level = max_level+1
			level_dict[level] = level_dict.get(level,0)+1
			attn_map=build_attention_map(root_node,max_level)
			avg_attn_map = np.sum(attn_map,axis=0)/np.diag(np.sum(attn_map!=0,axis=0))

			##### Build random  attention map 
			# height = set_height_bottomup(root_node)
			# max_level=set_level_topdown(root_node,0)
			# attn_map=build_attention_map_random(root_node)
			# avg_attn_map = attn_map





			avg_attn_map = avg_attn_map/np.sum(avg_attn_map,axis=0)
			# avg_attn_map=torch.nn.functional.softmax(torch.tensor(avg_attn_map))
			avg_attn_map = torch.tensor(avg_attn_map.transpose())
			out=avg_attn_map


			f = os.path.join(save_attnmap_folder, os.path.basename(bracket_file).replace('.bracket', '.attnmap'))
			torch.save(out, f)
			if idx%1000==0:
				print('%d/%d finished in the %s set.'%(idx,len(bracket_files),d))
				sys.stdout.flush()	
		# print(level_dict)


	