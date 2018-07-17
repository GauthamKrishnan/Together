import sys
# sys.path.append('/storage/home/gauthamk2512/.local/lib/python3.5/site-packages')
import numpy as np
import tensorflow as tf
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from graph import graph_coarsening, Graph
import os
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
		df = np.genfromtxt("../data/blog/edges.csv", delimiter = ',', dtype=int)
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
m = 2

epochs = 200
g = build_graph()
# g_prime = Graph(g.copy())
mat,g1 = graph_coarsening(g,m)

if sys.argv[1] == 'node2vec':
	node2vec = Node2Vec(g1, dimensions=128, walk_length=80, num_walks=10, workers=1) 
	model = node2vec.fit(window=10, min_count=1, batch_words=10)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
	model.wv.save_word2vec_format("embeds.csv")

elif sys.argv[1] == 'deepwalk':
	with open('edgelist.txt', 'w') as fp:
		for line in nx.generate_edgelist(g1, data=False):
			fp.write(line)
			fp.write('\n')
	os.system("deepwalk --input edgelist.txt --output embeds.csv --format edgelist --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10 --workers 1")

# Sort the Values
df = pd.read_csv('embeds.csv', delimiter = ' ', names = range(129))
df = df.sort_values(by=[0])
df.to_csv('embeds-sort.csv', sep=' ', header=None, index= False)

tf.reset_default_graph()
col = 128
row = mat[-1][0].shape[0]
embeddings = tf.placeholder(dtype = tf.float32, shape=[None,col], name = 'embeddings')
ground_truth = tf.placeholder(dtype = tf.float32, shape=[None,col], name = 'ground_truth')
# mask = tf.placeholder(dtype = tf.float32, shape=[None,col], name = 'mask')

matrix = tf.placeholder(dtype = tf.float32, shape = [None, None], name="orig_matrix")
matching = tf.placeholder(dtype = tf.float32, shape=[None,None], name="matching_matrix")
# lambdaa = tf.placeholder(dtype = tf.float32, shape=[], name="lambdaa")
lambdaa = 0.05
diagonal = get_diagonal(matrix)

approx_matrix, approx_diagonal = approx_matrix(matrix, diagonal ,lambdaa)
X = projection(matching, embeddings)

# theta_1 = tf.Variable(tf.truncated_normal(shape = [5,5],stddev=0.1))
# theta_2 = tf.Variable(tf.truncated_normal(shape = [5,5],stddev=0.1))

theta_1 = tanh_init([col, col], name="theta_1")
theta_2 = tanh_init([col, col], name="theta_2")

o_1 = layer_output(approx_matrix, approx_diagonal, X, theta_1)
o_2 = layer_output(approx_matrix, approx_diagonal, o_1, theta_2)

loss = tf.reduce_mean(tf.pow(ground_truth-o_2,2))
# loss = tf.losses.mean_squared_error(ground_truth, o_2)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# # Accuracy
# correct_pred = tf.equal(tf.argmax(o_2,1), tf.argmax(ground_truth,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
# accuracy = tf.metrics.accuracy(ground_truth, o_2)
saver = tf.train.Saver()
batch_size = 64 # If needed
with tf.Session() as sess:
	# Initializing the variables
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		
		# for i in range(1,5):
		
		df = np.genfromtxt('embeds-sort.csv', dtype = float, delimiter=' ', skip_footer=1) 
		nodes = df[:,0].astype(int)
		x_train = df[:,1:].astype(np.float32)
		# print(x_train)
		# print(x_train.shape)
		adj = mat[-1][0]
		# match = mat[-1][1]
		# print("shape", adj.shape[0])
		match = np.eye(adj.shape[0])
		# print("Begin")
		# cost_ = 0
		# for batch in range(num_batches):
			# mask_ = np.zeros_like(x_train)
			# get_rows = np.random.choice(range(x_train.shape[0]), batch_size, replace=False)
			# mask_[get_rows,:] = 1

		t1,t2,o=sess.run([theta_1, theta_2, optimizer],feed_dict={
			embeddings:x_train,
			ground_truth: x_train, 
			matrix : adj,
			matching: match,
			# mask: mask_

		})

		cost = sess.run(loss,feed_dict={
				embeddings:x_train,
				ground_truth: x_train, 
				matrix : adj,
				matching: match,
				# mask: mask_
		
		})
		# cost
	
		if epoch%10 == 0:
			print('Loss: {:.4f}'.format(cost))


	print("Theta 1\n", t1)
	print("Theta 2\n", t2)
	save_path = saver.save(sess, "tmp/model.ckpt")
	print("Model saved in path: %s" % save_path)


			


