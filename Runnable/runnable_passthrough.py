from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

'''Runnable passthrough is used to pass the input directly to the output without any processing.
This is useful when you want to use the input as it is without any modification.
'''
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)



prompt1= PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="explain the joke {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)
response = final_chain.invoke({"topic": "monkeys"})
print("Joke: ", response['joke'])
print("Explanation: ", response['explanation'])
