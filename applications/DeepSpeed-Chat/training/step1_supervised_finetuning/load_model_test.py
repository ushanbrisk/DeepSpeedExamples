from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "/ssd/output_test_202504091217_220k"
# model_name = "/home/luke/distributed_machine_learning/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/convert_model"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = '''A ship traveling along a river has covered $24 \mathrm{~km}$ upstream and $28 \mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \mathrm{~km}$ upstream and $21 \mathrm{~km}$ downstream, or half an hour more than for traveling $15 \mathrm{~km}$ upstream and $42 \mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly.

# Determine the speed of the ship in still water and the speed of the river.'''

# prompt = "Why do birds migrate south for the winter?"

# prompt = '''
# Prove that number $1$ can be represented as a sum of a finite number $n$ of real numbers, less than $1,$ not necessarily distinct, which contain in their decimal representation only the digits $0$ and/or $7.$ Which is the least possible number $n$?
# '''

# prompt = "Who was president of the United States in 1955?"

# prompt = "?????Ising Model"

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
