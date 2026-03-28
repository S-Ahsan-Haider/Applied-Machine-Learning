import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# Loading environement
load_dotenv()

# Setting up the lLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', 
                             temperature=0.3,
                             version='v1')

# Prompt template
template="""
You are an expert Code tutor based in India.
Your goal is to explain complex concepts to a high school 
level student who only knows Python programming using examples
from real life

Concept: {concept}
Difficulty Level: {level}

Explain the concept clearly in bullet points
"""
prompt_template = PromptTemplate(
    input_variables = ['concept', 'level'],
    template = template
)

formatted_prompt = prompt_template.format(concept="Docker vs Kubernetes", level="High School")
response = llm.invoke(formatted_prompt)

print("--- Here is your response ---")
print(response.content)