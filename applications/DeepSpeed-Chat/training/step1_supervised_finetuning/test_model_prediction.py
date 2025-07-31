from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import GenerationConfig

# model_name = "lukedai/Qwen2.5-1.5b-sft-ref"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",  # Automatically selects appropriate dtype (e.g., bfloat16, float16)
        device_map="cuda:1"    # Automatically distributes the model across available devices (GPUs)
    )

# model.generation_config = GenerationConfig(
#     use_cache=True  #
# )
#

#model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = '''## Task B-1.3.
#
# A ship traveling along a river has covered $24 \mathrm{~km}$ upstream and $28 \mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \mathrm{~km}$ upstream and $21 \mathrm{~km}$ downstream, or half an hour more than for traveling $15 \mathrm{~km}$ upstream and $42 \mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly.
#
# Determine the speed of the ship in still water and the speed of the river.'''


prompt = '''
On a board, the numbers from 1 to 2009 are written. A couple of them are erased and instead of them, on the board is written the remainder of the sum of the erased numbers divided by 13. After a couple of repetition of this erasing, only 3 numbers are left, of which two are 9 and 999. Find the third number.
'''

messages = [{"role": "user", "content": prompt}]

try:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
except:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

full_generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024  # Adjust as needed for desired output length
)


# generated_ids = [
#         output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
# ]
generated_ids = [
        output_ids[0:] for output_ids in full_generated_ids
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


