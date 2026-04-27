import random

from openai import OpenAI


client = OpenAI()

def get_embedding(text):
    return [random.random() for _ in range(1536)]