from datasets import load_dataset
import os
import hashlib
import torch
from deepspeed import get_accelerator


def create_student_teacher_dataset(local_rank,
                                    data_path,
                                    data_output_path,
                                    seed,
                                    teacher_tokenizer,
                                    student_tokenizer,
                                    max_seq_len):

    os.makedirs(data_output_path, exist_ok=True)
    fname = "_".join(data_path)
    teacher_tokenizer_name = teacher_tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    student_tokenizer_name = student_tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    #depends on tokenizer, as need tokenzier text
    fname = f"only_train_full_{fname}_tokenizer{teacher_tokenizer_name}_{student_tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{data_output_path}/traindata_{fname}.pt"

    cache_found = os.path.isfile(train_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank == 0 and (buf_create_cache.item() != 0):
        # copied from Distill proejct
        # dataset = load_from_disk("/ssd2/mlabonne_FineTome-100k")
        dataset = load_dataset(data_path[0], split="train") #here default to only 1 dataset
        dataset = dataset.shuffle(seed=seed)

        def sharegpt_format(example, teacher_tokenizer, student_tokenizer):
            conversations = example['conversations']
            message = []
            if isinstance(conversations, list):
                for conversation in conversations:
                    if isinstance(conversation, dict):
                        if conversation.get('from') == 'human':
                            message.append({"role": "user", "content": conversation.get('value', '')})
                        elif conversation.get('from') == 'gpt':
                            message.append({"role": "assistant", "content": conversation.get('value', '')})
                        elif conversation.get('from') == 'system':
                            message.insert(0, {"role": "system", "content": conversation.get('value', '')})
            if not any(msg.get('role') == 'system' for msg in message):
                message.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            student_text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            teacher_text = teacher_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            return {"student_text": student_text, "teacher_text": teacher_text}


        # Preprocess and tokenize the dataset

        map_kwargs = {}
        map_kwargs["num_proc"] = 52  # here is the parallel process number
        map_kwargs["desc"] = f"Applying chat template to {data_path[0]} dataset"

        print("Preprocessing and tokenizing dataset...")
        original_columns = dataset.column_names
        dataset = dataset.map(sharegpt_format,
                              fn_kwargs={"teacher_tokenizer": teacher_tokenizer, "student_tokenizer": student_tokenizer},
                              remove_columns=original_columns, **map_kwargs)


        def tokenize_function(examples, tokenizer, column_name):
            return tokenizer(examples[column_name], truncation=True, max_length=max_seq_len,
                             padding="max_length")


        teacher_tokenized_dataset = dataset.map(tokenize_function,
                                                fn_kwargs={"tokenizer": teacher_tokenizer, "column_name": "teacher_text"},
                                                batched=True,
                                                num_proc=8, remove_columns=["teacher_text"])
        teacher_tokenized_dataset = teacher_tokenized_dataset.rename_columns(
            {"input_ids": "teacher_input_ids", "attention_mask": "teacher_attention_mask"})

        student_tokenized_dataset = teacher_tokenized_dataset.map(tokenize_function, fn_kwargs={"tokenizer": student_tokenizer,
                                                                                                "column_name": "student_text"},
                                                                  batched=True,
                                                                  num_proc=8, remove_columns=["student_text"])
        student_tokenized_dataset = student_tokenized_dataset.rename_columns(
            {"input_ids": "student_input_ids", "attention_mask": "student_attention_mask"})

        print(f'finish read dataset')
        torch.save(student_tokenized_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname, weights_only=False)  # modifed 20250402 , weights_only=true

