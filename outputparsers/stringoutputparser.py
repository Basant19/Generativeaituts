from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os



load_dotenv()
hf_token=os.getenv ("HUGGINGFACEHUB_ACCESS_TOKEN")
LLM=HuggingFaceEndpoint (
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model=ChatHuggingFace (llm=LLM)


template1=PromptTemplate(
    template ='Write a detailed prompt on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template ='Write a 5 line summary on the following text /n {text}',
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'Artificial Intelligence'})
result1=model.invoke(prompt1)
prompt2=template2.invoke({'text':result1.content})
result2=model.invoke(prompt2)

print (result2.content)