from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda,RunnableBranch
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

prompt1 = PromptTemplate (
    template="Write a detail report {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate (
    template="Write a summary of the report on {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser ()
report_generator_chain = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split ())>50, RunnableSequence(prompt2, model, parser)), # RunnableSequence(prompt2, model, parser) use lcel prompt2|, model| parser no need to use function RunnableSequence
    RunnablePassthrough()

)

final_chain = RunnableSequence(report_generator_chain, branch_chain)
response = final_chain.invoke({"topic": "monkeys"})
print (response)