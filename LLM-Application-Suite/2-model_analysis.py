from sentence_transformers import SentenceTransformer
import time
import numpy as np

file_path = 'data.txt'

with open(file_path, 'r') as f:
    corpus = [line.strip() for line in f if line.strip()]

# Models for comparison:
# 1. larger, older
moda_1 = 'bert-base-nli-mean-tokens'

# 2. faster, smaller
moda_2 = 'all-MiniLM-l6-v2'


# MODEL 1: BERT based model  -----------------------------------------------

moda1 = SentenceTransformer(moda_1)

print('-'*50)
print("Name : ", moda_1)

# Size of vector it creates from 1st sentence only

corpus_embeddings_1 = moda1.encode(corpus[0])

vector_size_1 = corpus_embeddings_1.shape[0]        # How many no.s the model uses to represent the meaning of one sentence  (large = 764)

print("Size of the vector (Model 2): ", vector_size_1)
print('-'*50)




# MODEL 2: Newer, more efficient  ------------------------------------------

moda2 = SentenceTransformer(moda_2)

print('-'*50)
print("Name : ", moda_2)

# Size of vector it creates from 1st sentence only

corpus_embeddings_2 = moda1.encode(corpus[0])

vector_size_2 = corpus_embeddings_2.shape[0]        # How many no.s the model uses to represent the meaning of one sentence  (only half = 384)

print("Size of the vector (Model 2): ", vector_size_2)
print('-'*50)




# Comparing models' time for converting the entire corpus

print('-*-'*5, 'Model 1', '-*-'*5)

t1 = time.time()
corp_embed = moda1.encode(corpus, convert_to_tensor=False)
t2 = time.time()

print("Time taken: ", t2-t1, 'seconds')

# -----------------------------------------------------------------------

print('-*-'*5, 'Model 2', '-*-'*5)

t1 = time.time()
corp_embed = moda2.encode(corpus, convert_to_tensor=False)
t2 = time.time()

print("Time taken: ", t2-t1, 'seconds')