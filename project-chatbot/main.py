from fastapi import FastAPI, UploadFile
import numpy as np
import os

from utils import load_text,chunk_text
from embeddings import get_embedding
from retriever import VectorStore
from llm import ask_llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None

@app.post("/upload")
async def upload(file: UploadFile):
    global vector_store

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    chunks = chunk_text(text)

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = len(embeddings)

    vector_store = VectorStore(dim)
    vector_store.add(embeddings,chunks)

    return {"message": "Document processed"}

@app.get("/ask")
def ask(question: str):

    if vector_store is None:
        return {"error": "No documents uploaded yet"}
    query_embedding = get_embedding(question)
    relevant_chunks = vector_store.search(query_embedding)

    print("QUESTION:", question)
    print("CHUNKS:", relevant_chunks)
    context = "\n".join(relevant_chunks)
    answer = ask_llm(context,question)

    return {"answer": answer }

@app.on_event("startup")
def load_documents():
    global vector_store

    texts = []

    for filename in os.listdir("data"):
        with open(f"data/{filename}", "r", encoding="utf-8") as f:
            texts.append(f.read())

    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))

    embeddings = [get_embedding(chunk) for chunk in all_chunks]

    dim = len(embeddings[0])

    vector_store = VectorStore(dim)
    vector_store.add(embeddings, all_chunks)

    print("Knowledge base loaded ✅")