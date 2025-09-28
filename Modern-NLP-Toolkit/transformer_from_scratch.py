# 1. Tokenisation and Word Embeddings

import numpy as np

# Defining small vocabulary and a sentence
v = {"I": 0, "love": 1, "cats": 2, "and": 3, "dogs": 4}
sent = ['I','love','cats']
token_ids = [v[word] for word in sent]

# Creating random embedding matrix (5 words, 16 dimensions)
embedding_dim = 16
embedding_matrix = np.random.rand(len(v), embedding_dim)

# Lookup embeddings for the sentences
word_embeddings = np.array([embedding_matrix[token] for token in token_ids])
print()
print("Word Embeddings:")
print()
print(word_embeddings)

# 2. Positional Encodings

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos/np.power(10000, i/d_model)
            PE[pos, i] = np.sin(angle)
        if i+1 < d_model:
            PE[pos, i+1] = np.cos(angle)
    return PE

# Generating positional encodings for 3 words and 16 dimensions
pos_enc = positional_encoding(seq_len=3, d_model=embedding_dim)

# Adding a positional encoding to word embeddings
input_vectors = word_embeddings + pos_enc

print()
print("Input Vectors (Embeddings + Positional Encoding)")
print()
print(input_vectors)

# 3. Scaled Dot-Product Self-Attention
def softmax(x):
    e_x = np.exp(x-np.max(x))            # Stability trick
    return e_x / e_x.sum(axis=1, keepdims=True)

# a. Create random weight matrices for Q, K, W
W_Q = np.random.rand(embedding_dim, embedding_dim)
W_K = np.random.rand(embedding_dim, embedding_dim)
W_V = np.random.rand(embedding_dim, embedding_dim)

# b. Compute Q, K, W
Q = input_vectors @ W_Q
K = input_vectors @ W_K
V = input_vectors @ W_V

# c. Computing attention scores
dk = embedding_dim
scores = Q@K.T / np.sqrt(dk)          # Shape = (3,3)

# d. Appliying softmax to get attention weights
attention_weights = softmax(scores)

# e. Multiply attention weights with V 
output_vectors = attention_weights @ V

print('-'*50)
print("Attention Weights:\n", attention_weights)
print("Output Vectors:\n", output_vectors)

# 6. Encoder Block 

# a. Residual Connection
residual = output_vectors + input_vectors

# b. Feed Forward Network
W1 = np.random.rand(embedding_dim, embedding_dim)
b1 = np.random.rand(embedding_dim)
W2 = np.random.rand(embedding_dim, embedding_dim)
b2 = np.random.rand(embedding_dim)

def relu(x):
    return np.maximum(0, x)

ffn_output = relu(residual @ W1 + b1) @ W2 + b2

# c. Layer Normalization (simplified)
mean = ffn_output.mean(axis=-1, keepdims=True)
std = ffn_output.std(axis=-1, keepdims=True)
normalized_output = (ffn_output - mean) / (std + 1e-6)

print("Final Output Vectors:\n", normalized_output)




# 5. Visualisation and Analysis

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(attention_weights, annot=True, cmap="Blues", xticklabels=sent, yticklabels=sent)
plt.title("Attention Matrix")
plt.show()

# 







print('Done!')