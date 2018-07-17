import numpy as np
import networkx as nx
import scipy

def fetch_batch(graph, num_batches, batch_size = 64):

	matrix = nx.adjacency_matrix(graph)
	rows, cols = matrix.shape

	nodes = list(graph.nodes())
	n_nodes = len(nodes)
	batches = []
	for batch_id in range(num_batches):

		start = batch_id * batch_size
		if start >= n_nodes:
			break
			
		end = np.min([(batch_id+1) * batch_size, n_nodes])
		curr_bsize = end - start
		batch_density = curr_bsize / n_nodes

		node_ids = nodes[start:end]
		node_neighbors = matrix[start:end,:]
		#print(node_neighbors)
		# print(np.reshape(node_neighbors, (64,10312)))
		sum_neighbors = node_neighbors.sum(axis=1)
		mean_neighbor = (np.where(sum_neighbors != 0, node_neighbors/sum_neighbors, node_neighbors))
		batches.append((node_ids, mean_neighbor))
		#batches.append((node_ids, node_neighbors.todense()))
		

	return batches