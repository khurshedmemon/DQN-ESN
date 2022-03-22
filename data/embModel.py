import numpy as np
import graph_tool.all as gt
import os
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from config import get_config
#import node2vec
import graph.node2vec as n2vec
import pdb

class embModel:

	def __init__(self, g, config, emb_st):
		self.g = g
		self.p = config.p
		self.q = config.q
		self.emb_st = emb_st
		self.dimensions = config.dimensions
		self.walk_length = config.walk_length
		self.num_walks = config.num_walks
		self.window_size = config.window_size
		self.num_epoch = config.num_epoch
		self.workers = config.workers
		self.num_epoch = config.num_epoch
		self.eps = config.eps
		self.is_directed = config.is_directed		
		self.emb_output = "./data/" + config.env_name + ".emb"
		self.G = n2vec.Graph(g, self.is_directed, self.p, self.q)

	def learn_embeddings(self, walks):
		#walks = [map(str, walk) for walk in walks]
		walks = [list(map(str, walk)) for walk in walks]
		model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=self.workers, iter=self.num_epoch)
		model.wv.save_word2vec_format(self.emb_output)
		#print(model)		
		return model

	def generate_embeddings(self, g, emb_state, exp_nodes, nodes_to_embed):
		#pass the arguments
		self.emb_st	= emb_state
		self.G.preprocess_transition_probs(exp_nodes, nodes_to_embed)
		walks = self.G.simulate_walks(self.num_walks, self.walk_length, exp_nodes, nodes_to_embed)
		embed_model = self.learn_embeddings(walks)
		#shape = str(model.get_embedding().shape)		
		up_st_emb = self.read_embeddings(self.emb_output)
		print('updated state after embedding: ', up_st_emb)
		pdb.set_trace()
		#update the umbedding state
		self.emb_st = up_st_emb
		#X = sort_embeddings(model, gt_G)
		#return updated embeddings
		return up_st_emb


	#read embeddings
	def read_embeddings(self, emb_file):
		words = []
		vectors = []
		with open(emb_file, 'rb') as f:
			#data = file.readlines()[1:]			
			for line in f:
				fields = line.split()
				word = fields[0].decode('utf-8')
				#vector = np.fromiter((float(x) for x in fields[1:]), dtype=np.float)
				vector = [float(x) for x in fields[1:]]
				words.append(word)
				vectors.append(vector)
		#st = vectors
		st = np.array(vectors)
		return st