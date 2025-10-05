file_path = 'data.txt'

corpus = []

# read the lines (in data.txt) to the above empty list
# using 'with open' to ensure file closes automatically

with open(file_path, 'r') as f:
    
    # reading every line, removing extra spaces, new lines etc.
    corpus = [line.strip() for line in f if line.strip()]

print('-'*50)
print("Corpus is: ")
print(corpus)
print('-'*50)

# GENERATING EMBEDDINGS (machine readable vectors)
# model used - all-MiniLM-L6-v2, for lightweightedness and being fast

from sentence_transformers import SentenceTransformer 

moda = SentenceTransformer('all-MiniLM-L6-v2')

# GENERATING EMBEDDINGS FOR THE ENTIRE CORPUS
# encode() - takes string list and returns list of vectors (numpy arrays)

print('-'*50)
print("Generating embeddings ...")

corpus_embeddings = moda.encode(corpus, convert_to_tensor = False) 

print("Shape of embeddings ------> ")
print(corpus_embeddings.shape)               # 6 sentences, so something like (6,xxx) 
print('-'*50)
 


# INDEXING WITH FAISS 
# Like an organising library - to find things quicker (6, xxx) so easy, but (6 mil, xxx) would be not

import faiss
import numpy as np

# Ensuring embeddings are in numpy arrays
corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32)

# Dimension of the vectors y in (x,y)
D = corpus_embeddings.shape[1]

# creating a simple FAISS index -> IndexFlastL2 (L2 - Euclidean distance)

index = faiss.IndexFlatL2(D)

# adding the embeddings to the index
index.add(corpus_embeddings)

print('-'*50)

print('FAISS Index created with dimension D = ')
print(D)

print('-'*50)

print('Total vectors in index: ')
print(index.ntotal)

print('-'*50)


