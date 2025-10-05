from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os

# loading document
loader = TextLoader('data.txt')
docu = loader.load()

# Defining the splitter (cutting long book into small searchable chapters)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,                                
    chunk_overlap = 20,                  # allowing small overlap so that info loss don't happen at boundaries
    length_function = len 
)


# Splitting the document

chunks = text_splitter.split_documents(docu)

print('-'*50)
print('Total Original lines: ', len(docu))
print('Total chunks created: ', len(chunks))
print('Example of a chunk: ', chunks[0].page_content)
print('-'*50)

# Setting up the Retriever (vector database) ******************************************************************************************

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# need a way to wrap our query model (currently beuilt on SentenceTransformers)

from sentence_transformers import SentenceTransformer
# alternative to LangChain's wrapper, but will do for now


# Creating the embeddings object using the efficient model
embeddings_model = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")


# Creating the vector store (calculate and store embeddings for every chunk using the defined model)
vectorstore = Chroma.from_documents(documents=chunks,
                                    embedding = embeddings_model)

print("Vector Store created succesfully!")
print("Vector Store Size: ", vectorstore._collection.count())


# Setting up the generator (LLM, Flan-T5) *************************************************************************************************

# Importing HuggingFace wrapper for running local models
from langchain_community.llms import HuggingFacePipeline

# need pipeline components from transformers as well
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import torch            # for device configuration (selecting CPU vs GPU)

################
# Check if CUDA (NVIDIA GPU) is available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("*"*50)
print(f"DEVICE: Using {DEVICE} for LLM inference.")
print("*"*50)
################

# Simple Test Query
TEST_QUERY = "What technology is key for enterprise AI and what is Apple focusing on?"


# Define the small T5 model
model_name = "google/flan-t5-small"

# Loading model and Tokenizer from Hugging Face

toka = AutoTokenizer.from_pretrained(model_name)
moda = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Hugging face pipleine, defining the task of answering questions

pipe = pipeline(
    "text2text-generation",                    # Task for T5 model
    model = moda,
    tokenizer = toka, 
    max_new_tokens = 100,
    device= DEVICE                             # picking GPU if available
    )


# Wrapping pipleine in Langchain LLM Object, to allow it to talk to HuggingFace Object seamlessly

llm = HuggingFacePipeline(pipeline = pipe)

print('-'*50)
print(model_name, "is locked and loaded!!!")
print("Wuestion was: ", TEST_QUERY)
print('-'*50)


# Connecting Retriever (vector db) to Generator (LLM) ***********************************************************************************

# Orchestrator -> RetrievalQA chain, the single object that runs the entire RAG workflow automatically

from langchain.chains import RetrievalQA 

# creating final retrieval chain

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,                                 # The Generator
    retriever = vectorstore.as_retriever(),    # The Retriever
    chain_type = "stuff"
)

print('-'*50)
print("RAG QA Chain connected succesfully")
print()
print("Running QA Chain, will take some time while the LLM runs, so wait....")
result = qa_chain.invoke({'query': TEST_QUERY})

print('-'*10,"Final RAG answer",'-'*10)
print("Query: ", TEST_QUERY)
print("Answer: ", result['result'])
print('-'*50)