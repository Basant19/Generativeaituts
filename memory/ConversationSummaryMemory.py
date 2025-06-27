from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# Prompt Template
prompt = PromptTemplate.from_template(
    """The following is a conversation between a human and an AI assistant.
Summary of conversation:
{history}
Human: {input}
AI:"""
)

# Memory that summarizes chat history
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# Chain setup
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# Chat loop
print("\n🧠 SummaryMemory Chat. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = conversation.invoke({"input": user_input})
    print("🤖:", response["response"])
