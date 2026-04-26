from fastapi import FastAPI, UploadFile
import numpy as np

from utils import load_text,chunk_text
from embeddings import get_embdding
from retriever import VectorStore
from llm import ask_llm

app = FastAPI()

vector_store = None

@app.post("/upload")
async def upload(file: UploadFile):
    global vector_store

    content = await file.read()
    text = content.decode("utf-8")

    chunks = chunk_text(text)

    embeddings = [get_embdding(chunk) for chunk in chunks]

    dim = len(embeddings)

    vector_store = VectorStore(dim)
    vector_store.add(embeddings,chunks)

    return {"message": "Document processed"}

@app.get("/ask")
def ask(question: str);
    query_embedding = get_embdding(question)
    relevant_chunks = vector_store.search(query_embedding)

    context = "\n".join(relevant_chunks)
    answer = ask_llm(context,question)

    return {"answer": answer }