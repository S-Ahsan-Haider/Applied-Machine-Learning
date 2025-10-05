import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate

# ---(From Project 3) ----------------------------------------------------------------------------------------------------------------------------------------
# This section sets up the document loader, vector store, and QA chain.

# 1. Load Documents (Hardcoded to load a single 'data.txt' file in the current directory)
loader = TextLoader('data.txt')
docs = loader.load()

# 2. Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

# 3. Embeddings and Vector Store (Chroma)
# Note: Embeddings are intentionally set to 'cpu' to save VRAM for the main LLM.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory='chroma_db')

# 4. LLM Setup (Flan-T5)
model_name = "google/flan-t5-small"  # <-- REVERTED TO SMALL MODEL TO FIT MEMORY
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    device=DEVICE 
)

llm = HuggingFacePipeline(pipeline=pipe)

# 5. Define Prompt Template
template = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nAnswer:"
qa_prompt = PromptTemplate.from_template(template)

# 6. Create the Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": qa_prompt}
)
print(f"RAG QA Chain loaded on {DEVICE}.")


# ---(Project 4 starts now) ----------------------------------------------------------------------------------------------------------------------------------------
# This section sets up the agent that decides when to use the RAG tool.

# 7. Define the RAG Tool
# This wraps the RAG chain into a callable tool that the agent can execute.
rag_tool = Tool(
    name="RAG_Search",
    # The 'func' executes the qa_chain's invoke method and pulls the result.
    # The agent will use this tool to answer questions related to the 'data.txt' content.
    func=lambda x: qa_chain.invoke({'query': x})['result'], 
    description=(
        "Use this tool to answer specific questions based on internal document 'data.txt'. "
        "Always use this tool if the question refers to data within the document. "
        "Input should be the full question."
    )
)

# The same LLM (Flan-T5) is reused as the main reasoning engine for the agent.
agent_llm = llm 

# 8. Create and Run the Agent Executor
tools = [rag_tool]

# Initialize the Agent
# SWITCHED to STRUCTURED_CHAT_ZERO_SHOT_REACT, which can sometimes be more stable 
# with smaller models than the standard ReAct agent.
agent = initialize_agent(
    tools=tools, 
    llm=agent_llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT, 
    verbose=True, 
    handle_parsing_errors=True,
    # Limiting iterations prevents infinite loops if the model still struggles with formatting.
    max_iterations=4 
)

# Test the Agent

print("\n--- Testing Agent: Internal Knowledge Query ---")
# When the agent sees 'data.txt' in the query, it should use the RAG_Search tool.
agent.run("What are the main topics discussed in data.txt?")

print("\n--- Testing Agent: General Knowledge Query ---")
# The agent should answer directly without using the tool for this general query.
agent.run("What is the speed of light?")
