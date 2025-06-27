from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Set up the model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# Set up memory (stores full chat history)
memory = ConversationBufferMemory()

# Set up a conversation chain that uses memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Get user input
user_name = input("Enter your name: ")
intro_response = conversation.predict(input=f"Hi, my name is {user_name}")
print("\n🤖", intro_response)

# Ask for topic
topic = input("\nWhat topic would you like a summary of? ")
summary_prompt = f"Can you give me a detailed but simple summary of the topic '{topic}' for a person named {user_name}?"

# Generate summary
summary_response = conversation.predict(input=summary_prompt)
print("\n🧠 Summary for", user_name + ":\n")
print(summary_response)

# Optional: show memory buffer (conversation history)
print("\n📜 Conversation Memory:\n")
print(memory.buffer)
