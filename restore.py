import sys
# sys.path.append('/storage/home/gauthamk2512/.local/lib/python3.5/site-packages')
import numpy as np
import tensorflow as tf
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from graph import graph_coarsening, Graph
import scipy.io


def projection(matching, embeddings):
	return tf.matmul(matching, embeddings)

def get_diagonal(matrix):
	"""
	Returns
	-------
	The diagonal matrix of the given adj matrix
	"""
	degree_vector = tf.reduce_sum(matrix, 1)
	diagonal = tf.diag(degree_vector, name = 'diagonal')
	return diagonal

def approx_matrix(matrix, diagonal, lambdaa):
	"""
	Returns
	-------
	The approx adj matrix and its diagonal matrix
	"""

	adj_matrix = tf.add(matrix, tf.scalar_mul(lambdaa, diagonal))

	return adj_matrix, get_diagonal(adj_matrix)

def activation_function(value):
	return tf.tanh(value)

def layer_output(matrix, diagonal, X, theta):
	"""
	Returns
	-------
	The output of a conv layer
	"""
	d = tf.diag_part(diagonal)
	d_prime = tf.rsqrt(d)
	d_prime = tf.diag(d_prime)

	param = tf.matmul(d_prime, matrix)
	weighted_x = tf.matmul(X, theta)
	param = tf.matmul(param, d_prime)
	return activation_function(tf.matmul(param, weighted_x))

def tanh_init(shape, name=None, dtype=tf.float32, partition_info=None):
	init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def build_graph(max_rows = 1000, form='mat'):		
	g = Graph()

	if form == 'csv':
		df = np.genfromtxt("../data/edges.csv", delimiter = ',',max_rows=max_rows, dtype=int)
		for item in (df):
			g.add_edge(item[0],item[1],weight = 1)

	elif form == 'mat':
		file = scipy.io.loadmat('../data/wiki/wiki.mat')
		a =file['network'].tocoo()
		for r,c in zip(a.row,a.col):
			g.add_edge(r,c,weight=1)

	return g


"""
RUN THE NETWORK
"""

l = 0.05

# g = build_graph()

# print(g1.nodes)



# epochs = 200
g = build_graph()
# # g_prime = Graph(g.copy())
# mat,g1 = graph_coarsening(g,1)
mat,g1 = graph_coarsening(g,2)

# tf.reset_default_graph()
col = 128
row = mat[-1][0].shape[0]
embeddings = tf.placeholder(dtype = tf.float32, shape=[None,col], name = 'embeddings')
ground_truth = tf.placeholder(dtype = tf.float32, shape=[None,col], name = 'ground_truth')

matrix = tf.placeholder(dtype = tf.float32, shape = [None, None], name="orig_matrix")
matching = tf.placeholder(dtype = tf.float32, shape=[None,None], name="matching_matrix")
# lambdaa = tf.placeholder(dtype = tf.float32, shape=[], name="lambdaa")
lambdaa = 0.05
diagonal = get_diagonal(matrix)

approx_matrix, approx_diagonal = approx_matrix(matrix, diagonal ,lambdaa)
X = projection(matching, embeddings)

# theta_1 = tf.Variable(tf.truncated_normal(shape = [5,5],stddev=0.1))
# theta_2 = tf.Variable(tf.truncated_normal(shape = [5,5],stddev=0.1))

# theta_1 = tanh_init([col,col])
# theta_2 = tanh_init([col, col])
theta_1 = tf.get_variable("theta_1", shape=[col,col])
theta_2 = tf.get_variable("theta_2", shape=[col,col])


o_1 = layer_output(approx_matrix, approx_diagonal, X, theta_1)
o_2 = layer_output(approx_matrix, approx_diagonal, o_1, theta_2)

loss = tf.reduce_mean(tf.pow(ground_truth-o_2,2))
# loss = tf.losses.mean_squared_error(ground_truth, o_2)

#optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
	# Restore variables from disk.
	saver.restore(sess, "./tmp/model.ckpt")
	print("Model restored.")
	# # Check the values of the variables
	# print("o_1 : %s" % theta_1.eval())
	# print("o_2 : %s" % theta_2.eval())

	df = np.genfromtxt('embeds-sort.csv', dtype = float, delimiter=' ', skip_footer=1) 
	nodes = df[:,0].astype(int)
	embeds = df[:,1:].astype(np.float32)
	# print(x_train.shape)
	e = []
	o = []
	# print(o)
	for refine in range(2,len(mat)+1):
		adj = mat[-refine][0]
		match = mat[-refine][1]
		print("adj shape", adj.shape)
		print("match shape", match.shape)
		# match = np.eye(adj.shape[0])

		if (refine == 2):
			print(embeds.shape)
			e=sess.run(o_2,feed_dict={
			embeddings:embeds,
			matrix : adj,
			matching: match
			})
			# print(e)

		else:
			print(e)
			o=sess.run(o_2,feed_dict={
			embeddings:e,
			matrix : adj,
			matching: match
			})

	#print(o)
	np.savetxt("final_embed.csv",o,delimiter =' ')