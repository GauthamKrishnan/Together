import networkx as nx
import pandas as pd
import numpy as np
import math
from operator import itemgetter
from collections import defaultdict
#from node2vec import Node2Vec

class Graph(nx.Graph):

	def __init__(self, data=None, **attr):
		super(Graph, self).__init__(data, **attr)
		self.node2merge = defaultdict(list)

	# def __new__(cls):
	# 	if cls == nx.Graph:
	# 		return object.__new__(Graph)
	# 	return object.__new__(cls)

	# nx.Graph.__new__ = staticmethod(__new__)


	def adjacency_list(self):
		"""Return an adjacency list representation of the graph.

		The output adjacency list is in the order of G.nodes().
		For directed graphs, only outgoing adjacencies are included.

		Returns
		-------
		adj_list : lists of lists
			The adjacency structure of the graph as a list of lists.

		"""
		return list(map(tuple, iter(self.adj.values())))

	def get_matching_matrix(self, combine):
		"""
		Params
		-------
		combine: list of list of nodes to combine
		Returns 
		-------		
		The matching matrix of the graph
		"""
		rows = len(self.original_nodes)
		columns = len(combine)
		#print(self.original_nodes)
		matching =  np.zeros((rows,columns))

		for i,node in enumerate(self.original_nodes):
			row = np.array([1 if node in equivalent else 0 for equivalent in combine])
			matching[i] = row

		return matching

	
	def get_structural_matchings(self):
		"""
		Returns
		-------
		A set of vertices that are matched
		A list of list of nodes that are matched
		"""
		# print(self.adjacency_list())

		# Get the node list
		node_list = list(self.nodes())
		# Create a list, with element of form (node, neighbour_list)
		node_neighbour_list = [(node, tuple(self.neighbors(node))) for node in node_list]
		
		# Create a dictionary with keys as neighbour list and values as nodes having those neighbours
		neighbour2nodes = defaultdict(list)

		for neighbors in node_neighbour_list:
			neighbour2nodes[neighbors[1]].append(neighbors[0])

		matched = set()		# M1
		for matched_nodes in neighbour2nodes.values():
			if len(matched_nodes) > 1:
				# There are more than 1 nodes that share the same neighbour
				matched.update(matched_nodes)

		return matched, list(neighbour2nodes.values())

	def coarsen(self):
		"""
		Returns
		-------
		The matching matrix of the coarsened graph
		"""

		# Save the original nodes
		self.original_nodes = list(self.nodes())

		# Match according to structural equivalence
		matched, combine = self.get_structural_matchings()


		# TO MODIFY
		# Merges the nodes that are matched
		for i,node_combine in enumerate(combine):
			if(len(node_combine)>1):
				self.merger(node_combine)


		# print("Before Normalization")
		# print(list(self.edges(data='weight', default=1)))

		# print("Before Normalization")
		# print(list(self.edges(data='weight', default=1)))


			
		#Normalize all the edges of the graph
		for edge in list(self.edges()):
			self.get_heavy_edge(edge)


		# print("After Normalization")
		# print(list(self.edges(data='weight', default=1)))

		# print("After Normalization")
		# print(list(self.edges(data='weight', default=1)))


		# Sort the nodes according to the number of neighbors
		node_list = sorted(self.nodes(), key = lambda x: len(tuple(self.neighbors(x))))
		# print("node-list", node_list)
		for v in node_list:
			# Sort the neighbours accoridng to their weight
			neighbour_list = sorted(list(self.neighbors(v)), key = lambda x: self[v][x]['weight'], reverse = True)

			for u in neighbour_list :
				if u in matched or v in matched or v == u:
					continue
				else:
					combine.remove([v]) 
					combine.remove([u]) 
					matched.update([v,u])
					combine.append([v,u])

		# print("matched")
		# print(matched)
		
		sorted_combine = sorted(combine, key=itemgetter(0))
		matching_matrix = self.get_matching_matrix(sorted_combine)
		# print(matching_matrix)
		return matching_matrix

	def get_matchinglist1(self):
		
		matching_list_prelim = get_structural_matchings()

	def are_neighbours(self, node_1, node_2):
		return self.has_edge(node_1, node_2)

	def get_heavy_edge(self, org_edge):

		node_1 = (org_edge)[0] 
		node_2 = (org_edge)[1]
		edge_weight = self.get_edge_data(node_1,node_2)['weight']
		deg_1 = self.degree(node_1)
		deg_2 = self.degree(node_2)

		norm_weight = edge_weight / math.sqrt(deg_1 * deg_2)
		self[node_1][node_2]['weight'] = norm_weight

	def merger(self, nodes):
		"""
		Takes the list of nodes to combine and combines nodes 2:n to node 1
		returns the new_node
		"""
		new_node = nodes[0]

		for node in nodes[1:]:
			for neighbor in self.neighbors(node):
				if self.has_edge(new_node,neighbor):
					self[new_node][neighbor]['weight'] += self[node][neighbor]['weight']

				else:
					self.add_edge(new_node,neighbor, weight = 1)

		for n in list(nodes[1:]):
			self.remove_node(n)

		self.node2merge[new_node] = nodes

		return new_node


	def adjacency_matrix(self,adj_mat,match_mat):

		temp = np.matmul(match_mat.T , adj_mat)
		adj = np.matmul(temp,match_mat)
		return adj 
	

def graph_coarsening(g, m=1):

	i = 0
	mat = []
	g1 = Graph()
	while(i <= m):
		g1 = g.copy()
		matrix = nx.to_numpy_matrix(g).copy()
		a = np.array(matrix)
		b = np.array(g.coarsen()).copy()
		# print("Adjacency Matrix:")
		new_mat =  (g.adjacency_matrix(a,b))
		g.clear()
		g = Graph(new_mat)
		mat.append((a,b))
		i = i + 1

	return mat,g1

def graph_coarsening_list(g, m=1):

	i = 0
	mat = []
	g1 = Graph()
	while(i <= m):
		g1 = g.copy()
		matrix = nx.to_numpy_matrix(g).copy()
		a = np.array(matrix)
		b = np.array(g.coarsen()).copy()
		# print("Adjacency Matrix:")
		new_mat =  (g.adjacency_matrix(a,b))
		g.clear()
		g = Graph(new_mat)
		mat.append((a,b))
		i = i + 1

	return mat,g1


if __name__ == "__main__":
		
	# 


	g = Graph()
	# #df = pd.read_csv("C:\\Users\\Krishnan\\Desktop\\BlogCatalog-dataset\\data\\edges.csv")
	# #edge_list = list(df.values)

	# #for index,item in enumerate(edge_list):
	# 	#if index <= 50:
	# 		#g.add_edge(edge_list[index][0],edge_list[index][1],weight = 1)

	# mat = list()
	# g.add_edge(1,2,weight = 1)
	# g.add_edge(2,3,weight = 1 )
	# g.add_edge(1,3,weight = 1)
	# g.add_edge(1,4,weight = 1)
	# g.add_edge(1,5,weight = 1)
	# print(g.edges)
	# matrix = nx.to_numpy_matrix(g)
	# print(matrix)
	# graph_coarsening(g,1)
	# matrix = nx.to_numpy_matrix(g)
	# print(matrix)
	# a = np.array(matrix)
	# b = np.array(g.coarsen())
	# print(g.adjacency_matrix(a,b))
	# print("hi")
	# print(mat)



	# # Precompute probabilities and generate walks
	# node2vec = Node2Vec(g, dimensions=10, walk_length=7, num_walks=25, workers=1) 

	g = Graph()
	df = pd.read_csv("C:\\Users\\Krishnan\\Desktop\\BlogCatalog-dataset\\data\\edges.csv")
	edge_list = list(df.values)

	for index,item in enumerate(edge_list):
		if index <= 50:
			g.add_edge(edge_list[index][0],edge_list[index][1],weight = 1)

	mat = list()
	#g.add_edge(1,2,weight = 1)
	#g.add_edge(2,3,weight = 1 )
	#g.add_edge(1,3,weight = 1)
	#g.add_edge(1,4,weight = 1)
	#g.add_edge(1,5,weight = 1)
	print(g.edges)
	matrix = nx.to_numpy_matrix(g)
	print(matrix)
	graph_coarsening(g,2)
	#matrix = nx.to_numpy_matrix(g)
	#print(matrix)
	#a = np.array(matrix)
	#b = np.array(g.coarsen())
	#print(g.adjacency_matrix(a,b))

	# Precompute probabilities and generate walks
	node2vec = Node2Vec(g, dimensions=10, walk_length=7, num_walks=25, workers=1) 

	# Embed
	model = node2vec.fit(window=4, min_count=1, batch_words=5)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

	# Look for most similar nodes
	#model.wv.most_similar('2')  # Output node names are always strings

	# Save embeddings for later use
	model.wv.save_word2vec_format("nodesrw.csv")


	# Save model for later use
	model.save("nodes2vec.csv")


