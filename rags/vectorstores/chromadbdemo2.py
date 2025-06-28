import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load .env variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Step 1: Set up the LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm=llm)

# Step 2: Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Clear old Chroma DB to avoid stale data and missing metadata
db_path = "movie_vector_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# Step 4: Define sample movie documents with genre metadata
docs = [
    Document(page_content="Inception is a science fiction and action film directed by Christopher Nolan, where a thief enters people's dreams to steal secrets.",
             metadata={"movie": "Inception", "genre": "Science Fiction, Action"}),

    Document(page_content="Titanic is a romantic drama directed by James Cameron, depicting the tragic sinking of the Titanic ship.",
             metadata={"movie": "Titanic", "genre": "Romance, Drama"}),

    Document(page_content="The Shawshank Redemption is a drama film based on a novella by Stephen King, known for its themes of hope and friendship.",
             metadata={"movie": "Shawshank Redemption", "genre": "Drama"}),

    Document(page_content="The Dark Knight is a superhero and action film directed by Christopher Nolan, featuring Batman's battle against the Joker.",
             metadata={"movie": "The Dark Knight", "genre": "Superhero, Action"}),

    Document(page_content="Interstellar is a science fiction film that explores space travel through wormholes and black holes in search of a new home for humanity.",
             metadata={"movie": "Interstellar", "genre": "Science Fiction, Adventure"}),
]

# Step 5: Create Chroma vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=db_path,
    collection_name="movie_info"
)
vector_store.add_documents(docs)

# Step 6: Inform user
print("🎬 Available Movie Topics:\n- Inception\n- Titanic\n- The Shawshank Redemption\n- The Dark Knight\n- Interstellar")
print("📂 Genres Available: Science Fiction, Action, Romance, Drama, Superhero, Adventure")

# Step 7: Take user query
query = input("\n❓ Ask a question about these movies (e.g., 'Which movies are science fiction?' or 'Which movies have similar genres?'): ")

# Step 8: Retrieve relevant documents
results_with_score = vector_store.similarity_search_with_score(query=query, k=3)

# Step 9: Show results with metadata safely
print("\n🔍 Top Matching Documents:")
for i, (doc, score) in enumerate(results_with_score, 1):
    movie = doc.metadata.get('movie', 'Unknown')
    genre = doc.metadata.get('genre', 'N/A')
    print(f"{i}. 🎞️ Movie: {movie}")
    print(f"   📄 {doc.page_content}")
    print(f"   🏷️ Genre: {genre}")
    print(f"   📊 Similarity Score: {score:.4f} (lower is better)\n")

# Step 10: Prepare multi-doc context for RAG-style answer
combined_context = "\n\n".join([doc.page_content for doc, _ in results_with_score])
rag_prompt = f"""You are a movie expert. Based on the following documents, answer this question:
{combined_context}

Question: {query}
"""

# Step 11: Get LLM answer
response = model.invoke(rag_prompt)
print("🤖 LLM Answer:")
print(response.content)
