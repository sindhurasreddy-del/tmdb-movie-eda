from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from groq import Groq

# ── What: Load the embedding model
# ── Why: This converts text to vectors (numbers that capture meaning)
#         'all-MiniLM-L6-v2' is small, fast, and free — perfect for learning
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Done!")

# ── What: Load movie data
# ── Why: This is the knowledge base our chatbot will search through
print("Loading movie data...")
df = pd.read_csv('tmdb_5000_movies.csv')
df = df[['title', 'overview', 'genres', 'release_date', 'budget', 'revenue']].dropna(subset=['overview'])
df = df.head(500)  # use first 500 movies to keep it fast
print(f"Loaded {len(df)} movies")

# ── What: Create text chunks from each movie
# ── Why: We combine title + overview into one string per movie
#         This is what gets embedded and searched
print("Creating movie descriptions...")
movie_texts = []
for _, row in df.iterrows():
    text = f"Title: {row['title']}. Description: {row['overview']}"
    movie_texts.append(text)

# ── What: Convert all movie texts to embeddings
# ── Why: FAISS needs vectors (numbers), not text
#         Each movie becomes a list of 384 numbers capturing its meaning
print("Creating embeddings (this may take 1-2 minutes)...")
embeddings = embedder.encode(movie_texts, show_progress_bar=True)
print(f"Created {len(embeddings)} embeddings of size {embeddings.shape[1]}")

# ── What: Build FAISS index
# ── Why: FAISS is a special database optimized for similarity search
#         It can find the most similar vectors in milliseconds
print("Building vector store...")
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(embeddings.astype('float32'))
print(f"Vector store built with {index.ntotal} movies")

# ── What: Connect to Groq (Llama 3)
# ── Why: This is our LLM that will answer questions using retrieved context
client = Groq(api_key="your_gsk_key_here")

def ask_movies(question):
    """
    RAG pipeline:
    1. Embed the question
    2. Search FAISS for similar movies
    3. Send context + question to LLM
    4. Return answer
    """
    
    # Step 1: Embed the question
    # Why: We need to convert the question to a vector
    #      so we can compare it to movie vectors
    question_embedding = embedder.encode([question])
    
    # Step 2: Search FAISS for top 5 most relevant movies
    # Why: We find the 5 movies whose meaning is closest to the question
    #      D = distances, I = indices of matching movies
    D, I = index.search(question_embedding.astype('float32'), k=5)
    
    # Step 3: Collect the relevant movie texts
    # Why: These become the "context" we give to the LLM
    context_movies = []
    for idx in I[0]:
        context_movies.append(movie_texts[idx])
    
    context = "\n\n".join(context_movies)
    
    # Step 4: Send to Llama 3 with context
    # Why: This is the "Augmented Generation" part of RAG
    #      We're giving the LLM relevant information it wouldn't otherwise have
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful movie expert assistant. 
                Answer questions using the provided movie context.
                Be concise and specific."""
            },
            {
                "role": "user",
                "content": f"""Context (relevant movies from our database):
{context}

Question: {question}

Answer based on the context provided above."""
            }
        ]
    )
    
    return response.choices[0].message.content

# ── Test the RAG chatbot with 3 questions
print("\n" + "="*50)
print("RAG Movie Chatbot Ready!")
print("="*50)

questions = [
    "tell me about BTS",
    "What movies involve superheroes?",
    "Which movies are about space exploration?"
]

for q in questions:
    print(f"\nQuestion: {q}")
    print(f"Answer: {ask_movies(q)}")
    print("-"*40)
