from openai import OpenAI

client = OpenAI()

def get_embdding(text):
    response = client.embeddings.create( model = "text-embedding-3-small",
                                        input=text)
    return response.data[0].embedding