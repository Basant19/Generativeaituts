from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
hf_token=os.getenv ("HUGGINGFACEHUB_ACCESS_TOKEN")
LLM=HuggingFaceEndpoint (
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
prompt =PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)


model=ChatHuggingFace (llm=LLM)

parser=StrOutputParser()

chain=prompt|model|parser
result=chain.invoke({'topic':'black hole'})
print (result)


'''
to visualize the chain 
chain.get_graph().print_ascii
'''
