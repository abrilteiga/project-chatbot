from openai import OpenAI

#create client to interact with the  OpenAI API
client = OpenAI()

def ask_llm(context, question):
    prompt = f"Answer the question using ONLY the context below \n Context: {context} \n Question:{question} \n"

    # I choose the gpt-4.1-mini for my model
    response = client.chat.completions.create(model = "gpt-4.1-mini", messages = [{"role" : "user", "content" : prompt}])

    # Return the model’s text output
    return response.choices[0].message.content