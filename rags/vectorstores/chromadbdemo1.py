from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Step 1: Set up the LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm=llm)

# Step 2: Embedding model for converting text to vector
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create sample documents (Planet facts)
docs = [
    Document(page_content="Mercury is the closest planet to the Sun and has no atmosphere.", metadata={"planet": "Mercury"}),
    Document(page_content="Venus has a thick atmosphere and is the hottest planet.", metadata={"planet": "Venus"}),
    Document(page_content="Earth is the only planet known to support life.", metadata={"planet": "Earth"}),
    Document(page_content="Mars is called the Red Planet and might have had water in the past.", metadata={"planet": "Mars"}),
    Document(page_content="Jupiter is the largest planet and has a big red storm.", metadata={"planet": "Jupiter"}),
]

# Step 4: Store documents in Chroma vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="planet_vector_db",
    collection_name="planet_info"
)

vector_store.add_documents(docs)

# Step 5: Ask a question related to the documents
query = "Which planet has a thick atmosphere and is very hot?"

# Step 6: Retrieve relevant documents from vector store
relevant_docs = vector_store.similarity_search(query=query, k=1)

print("\n🔍 Top Matching Document:")
print(relevant_docs[0].page_content)

# Step 7: Use the LLM to answer the question using that info
rag_prompt = f"Answer this question based on the document:\nDocument: {relevant_docs[0].page_content}\nQuestion: {query}"
response = model.invoke(rag_prompt)

print("\n🤖 LLM Answer:")
print(response.content)
