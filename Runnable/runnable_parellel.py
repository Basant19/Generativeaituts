from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

model1= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model2 = ChatHuggingFace(llm=llm)

prompt1= PromptTemplate(
    template="Generate a tweet on the topic {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Generate a LinkedIn post on the topic {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()

parellel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1, model1, parser),
    'linkedin':RunnableSequence(prompt2, model2, parser)
})

response = parellel_chain.invoke({"topic": "AI in 2024"})
print("Output by gemini \n",response['tweet'])
print("Output by mistral \n",response['linkedin'])