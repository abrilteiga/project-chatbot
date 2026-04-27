from openai import OpenAI

#create client to interact with the  OpenAI API
client = OpenAI()

def ask_llm(context, question):
   prompt = f"""You are a kind, supportive assistant helping young girls understand puberty and periods. Use simple language. Be reassuring and positive. Do not give medical diagnoses.

Context:
{context}

Question:
{question}
"""

    # I choose the gpt-4.1-mini for my model
    response = client.chat.completions.create(model = "gpt-4.1-mini", messages = [{"role" : "user", "content" : prompt}])

    # Return the model’s text output
    return response.choices[0].message.content