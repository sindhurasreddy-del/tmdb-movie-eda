from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from groq import Groq
import os

# ── What: Create the FastAPI app
# ── Why: This is the main object that handles all HTTP requests
app = FastAPI(title="Movie RAG API", version="1.0")

# ── What: Load everything at startup
# ── Why: We load models once when the server starts, not on every request
#         Loading a model takes 10 seconds — we don't want that per request

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading movie data...")
df = pd.read_csv('tmdb_5000_movies.csv')
df = df[['title', 'overview', 'genres', 'release_date', 'budget', 'revenue']].dropna(subset=['overview'])
df = df.head(500)

movie_texts = []
for _, row in df.iterrows():
    text = f"Title: {row['title']}. Description: {row['overview']}"
    movie_texts.append(text)

print("Creating embeddings...")
embeddings = embedder.encode(movie_texts)

print("Building vector store...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("API ready!")

# ── What: Define the input shape using Pydantic
# ── Why: FastAPI uses this to validate incoming requests automatically
class Question(BaseModel):
    text: str  # the question must be a string

# ── What: Health check endpoint
# ── Why: Standard practice — lets you verify the server is running
#         Used by Docker, cloud platforms, and monitoring tools
@app.get("/")
def root():
    return {"status": "Movie RAG API is running!"}

# ── What: The main endpoint — ask a question, get an answer
# ── Why: POST because we're sending data (the question) to the server
@app.post("/ask")
def ask_question(question: Question):
    
    # Step 1: Embed the question
    question_embedding = embedder.encode([question.text])
    
    # Step 2: Search for relevant movies
    D, I = index.search(question_embedding.astype('float32'), k=5)
    
    # Step 3: Build context from retrieved movies
    context_movies = [movie_texts[idx] for idx in I[0]]
    context = "\n\n".join(context_movies)
    
    # Step 4: Ask Llama 3 with context
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful movie expert. Answer questions using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question.text}"
            }
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Step 5: Return structured response
    return {
        "question": question.text,
        "answer": answer,
        "sources": [movie_texts[idx][:100] + "..." for idx in I[0][:3]]
    }

# ── What: Get list of available movies
# ── Why: Useful endpoint for the frontend to show what's available
@app.get("/movies")
def get_movies():
    return {
        "total": len(df),
        "sample": df['title'].head(10).tolist()
    }