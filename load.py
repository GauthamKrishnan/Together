import numpy as np 
import pandas as pd 
from gensim.models import Word2Vec, KeyedVectors

rows = np.arange(10312)
df = pd.read_csv('epbembed_blog.csv', header =None, delimiter = ' ')
df = pd.DataFrame(index = rows, data = df)
df.to_csv('embeds-finalblog.csv', sep = ' ', header=None)
print(df.shape)