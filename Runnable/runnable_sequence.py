from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv ()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint (
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)



prompt=PromptTemplate (
    template="write a short story on the topic {topic}",
    input_variables=["topic"]
)

model= ChatHuggingFace (llm=llm)
parser=StrOutputParser ()

chain =RunnableSequence(prompt,model,parser)

response = chain.invoke({"topic": "A day in the life of a cat"})
print(response)