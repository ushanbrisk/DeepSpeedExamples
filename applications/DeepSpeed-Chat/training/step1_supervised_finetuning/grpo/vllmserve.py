from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Round 1
# messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
messages = [{"role": "user", "content": "A ship traveling along a river has covered $24 \mathrm{~km}$ upstream and $28 \mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \mathrm{~km}$ upstream and $21 \mathrm{~km}$ downstream, or half an hour more than for traveling $15 \mathrm{~km}$ upstream and $42 \mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly. \
Determine the speed of the ship in still water and the speed of the river."}]


# For granite, add: `extra_body={"chat_template_kwargs": {"thinking": True}}`
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content:", reasoning_content)
print("content:", content)