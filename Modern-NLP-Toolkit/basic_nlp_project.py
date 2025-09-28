import nltk
from nltk.tokenize import word_tokenize

# Download a necessary NLTK resource the first time you run
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# A simple sentence
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text into individual words
tokens = word_tokenize(text)

# Print the tokens
print("Original Text:", text)
print("Tokens:", tokens)