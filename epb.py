import sys
# sys.path.append('/storage/home/gauthamk2512/.local/lib/python3.5/site-packages')
import networkx as nx
import numpy as np
import random
import tensorflow as tf
import batch
import scipy.io 
from gensim.models import Word2Vec, KeyedVectors

"""
Custom Graph
"""

class Graph(nx.Graph):
	def __init__(self):
		super(Graph, self).__init__()


"""
Build Graph
"""

def build_graph(filename, max_rows = 100):		
	g = Graph()

	form = filename.split('.')[-1]

	if form == 'csv':
		df = np.genfromtxt(filename, delimiter = ',', dtype=int)
		for item in (df):
			g.add_edge(item[0],item[1],weight = 1)

	elif form == 'mat':
		file = scipy.io.loadmat(filename)
		a =file['network'].tocoo()
		for r,c in zip(a.row,a.col):
			g.add_edge(r,c,weight=1)

	return g


g = build_graph(filename = '../data/blog/edges.csv')

print("Graph is Built")

"""
Custom Functions
"""

# Returns the embeddings of nodes of the current batch
def get_embedding(embeddings, node):
	
	return tf.gather(params = embeddings, indices = node)

# Returns the mean of the neighbor embeddings of the current batch
def get_mean_embedding(embeddings, neighbors):
	
	return tf.matmul(neighbors, embeddings)


#def loss_fn(embeddings,neighbors_embeddings, negative_embeddings, batch_size, gamma):
def loss_fn(gamma):
	
	# Get the embeddings of the sampled nodes
	embeds = get_embedding(embeddings, batch_nodes)
	neighbors_embeddings = get_mean_embedding(embeddings, neighbors)
	negative_embeddings = get_embedding(embeddings, negative_sample)
	
	d1 = tf.sqrt(tf.add(tf.reduce_sum(tf.square(tf.subtract(neighbors_embeddings,embeds)),axis = 1),0.000001))
	#d1 = tf.sqrt(tf.add(tf.reduce_sum(tf.square(tf.subtract(get_neighbors,get_embeds)),axis = 1),0.000001))
	#d1 = tf.square(tf.subtract(embeddings, neighbours_embeddings))
	#d1 = tf.reduce_sum(d1, axis = 1)
	# d1 = tf.cond(tf.reduce_all(d1>0.1),lambda: tf.sqrt(d1), lambda: d1)

	d2 = tf.sqrt(tf.add(tf.reduce_sum(tf.square(tf.subtract(neighbors_embeddings,negative_embeddings)),axis = 1),0.000001))
	#d2 = tf.sqrt(tf.add(tf.reduce_sum(tf.square(tf.subtract(get_neighbors,get_negatives)),axis = 1),0.000001))
	#d2 = tf.square(tf.subtract(shuffled_embeddings, neighbours_embeddings))
	#d2 = tf.reduce_sum(d2, axis = 1)
	# d2 = tf.cond(tf.reduce_all(d2>0.1),lambda: tf.sqrt(d2), lambda: d2) 

	diff = tf.subtract(d1,d2)
	biased_diff = tf.add(diff,gamma)
	#pos_part = tf.clip_by_value(biased_diff,0.0,5.0)
	pos_part = tf.nn.relu(biased_diff)
	loss = tf.reduce_sum(pos_part)

	return loss
	#return (loss/batch_size)


"""
Variables and Placeholders
"""
embeddings = tf.Variable(embeds, name = "embeddings")
batch_nodes = tf.placeholder(dtype = tf.int32, name="batch_nodes")
batch_size = tf.placeholder(dtype = tf.float32, shape=[], name = "batch_size")
neighbors = tf.placeholder(dtype = tf.float32, name='neighbors')
negative_sample = tf.placeholder(dtype = tf.int32, name = 'negative_sample')


nodes = len(g.nodes())
dim = 128
gamma = 1.0 # Value depends on the dataset
num_samples = 5 # If more than one negative sample needed 

embeds = tf.random_uniform(
    shape = [nodes, dim],
    dtype=tf.float32,
    minval= -0.0884,
    maxval= 0.0884
)
'''
embeds = tf.truncated_normal(
    shape = [nodes, dim],
    dtype=tf.float32,
)
'''

"""
Build Network
"""

#Get the embeddings of the sampled nodes
#neighbours_embeddings = get_mean_embedding(embeddings, neighbors)
#neg_embed = get_embedding(embeddings, negative_sample)
#pos_embed = get_embedding(embeddings, batch_nodes)

# Loss
#loss = loss_fn(pos_embed,neighbours_embeddings, neg_embed, batch_size, gamma)
loss = loss_fn(gamma)

# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)

#saver = tf.train.Saver()

num_batch = 162 # Value Depends on dataset used. 
batch_s = 64 # Default

batches = batch.fetch_batch(g, num_batch, batch_s)

# No. of epochs
epochs = 200

with tf.Session() as sess:
	
	# Initializing the variables
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		cost_ = 0
		#neg_sample = np.random.choice(nodes,1).astype(np.int32)
		for batch_id in batches:
			
			#for sample in range(num_samples):
			sample_size = len(batch_id[0])
			neg_sample = np.random.choice(nodes, sample_size).astype(np.int32)
			
			sess.run(optimizer, feed_dict = {
				batch_nodes: batch_id[0],
				neighbors: batch_id[1],
				negative_sample: neg_sample,
				batch_size: batch_s
				})

			cost = sess.run(loss,feed_dict = {
				batch_nodes: batch_id[0],
				neighbors: batch_id[1],
				negative_sample: neg_sample,
				batch_size: batch_s
				})
			
			cost_ = cost_ + cost

		if (epoch % 10) == 0 :
			print("Loss: ", cost_/num_batch)

	embeds = sess.run(embeddings,{})
	
	# Save the final embeddings in csv file
	np.savetxt('epbembed_blog.csv',embeds, delimiter = ' ')
	#save_path = saver.save(sess, "tmp/model-epb.ckpt")
	#print("Model saved in path: %s" % save_path)

# Prepare the final embeddings in a format suitable for gensim package
rows = np.arange(10312)
df = pd.read_csv('epbembed_blog.csv', header =None, delimiter = ' ')
df = pd.DataFrame(index = rows, data = df)
df.to_csv('embeds-finalblog.csv', sep = ' ', header=None)

# CSV file ready to send scoring.py for evaluation









	

		
