from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load Hugging Face API key
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# HuggingFaceEndpoint LLM setup
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

'''
Runnables are composable building blocks in LangChain. They allow you to build data processing pipelines (or chains) by linking various components together.
A abstract class which is created which  classes  like PromptTemplate, ChatHuggingFace, StrOutputParser, etc. inherit from it.

Each Runnable:
Can be invoked with .invoke(input)
Can be composed using | (pipe operator)
Can run asynchronously with .ainvoke()
Can be mapped to datasets using .batch(), .stream(), etc.
'''

# Wrap in a Chat model
model = ChatHuggingFace(llm=llm)

# Step 1: Create a PromptTemplate (this is also a Runnable)
prompt = PromptTemplate.from_template("What is the capital of {country}?")

# Step 2: Create an OutputParser to extract string output (Runnable)
parser = StrOutputParser()

# Step 3: Chain Prompt → Model → Parser using `|` (this forms a Runnable chain)
chain = prompt | model | parser

# Step 4: Run the chain (invoke the full pipeline)
response = chain.invoke({"country": "India"})

# Output the result
print(response)
