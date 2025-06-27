from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv ()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint (
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate (
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser ()
joke_generator_chain = RunnableSequence(prompt, model, parser)

def wordcount (text):
    return len(text.split())

parellel_chain =RunnableParallel ({
    'joke':RunnablePassthrough(),
    'wordcount': RunnableLambda (wordcount)
})

final_chain = RunnableSequence(joke_generator_chain, parellel_chain)
response = final_chain.invoke({"topic": "monkeys"})
print (response)