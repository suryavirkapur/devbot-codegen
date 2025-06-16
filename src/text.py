from openai import OpenAI

client = OpenAI()


def generate_text(prompt: str) -> str:
    response = client.responses.create(model="gpt-4.1", input=prompt)
    print(response.output_text)
    return response.output_text
