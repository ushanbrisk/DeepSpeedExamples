from pipelayers_grpo import test_load_model,convert_model_to_hf_qwen25_3b
new_folder = '/ssd2/output_test_202506201544/global_step2144'
output_dir = '/ssd2/output_test_202506201544'
convert_model_to_hf_qwen25_3b(new_folder, output_dir)

model, tokenizer = test_load_model(output_dir)
# model = model.to(device)
for name, param in model.named_parameters():
    print(f"name: {name}")

#this file is only to collect weight from state_dict files
#it first collect all layers weight files into .bin files
#then read from disk this bin file


# from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import HfFolder
#
# # Replace with your model and tokenizer
# model_name = "your-model-name"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Replace with your repository name
# repo_id = "your-username/your-model-name"
#
# # Log in to Hugging Face Hub
# HfFolder.save_token("YOUR_HUGGING_FACE_TOKEN") # Replace with your token
#
# # Push the model and tokenizer to the Hub
# model.push_to_hub(repo_id, use_auth_token=True)
# tokenizer.push_to_hub(repo_id, use_auth_token=True)