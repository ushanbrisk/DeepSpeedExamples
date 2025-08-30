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


#only for openr1-220k dataset
def create_student_teacher_dataset_220k(local_rank,
                                    data_path,
                                    data_output_path,
                                    seed,
                                    teacher_tokenizer,
                                    student_tokenizer,
                                    max_seq_len,
                                    filter_answer_len = 5000):

    os.makedirs(data_output_path, exist_ok=True)
    fname = "_".join(data_path)
    teacher_tokenizer_name = teacher_tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    student_tokenizer_name = student_tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    #depends on tokenizer, as need tokenzier text
    fname = f"only_train_full_{fname}_tokenizer{teacher_tokenizer_name}_{student_tokenizer_name}_seqlen{max_seq_len}_filterlen{filter_answer_len}"
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
        dataset = dataset.filter(lambda x: len(x['messages'][1]['content']) < filter_answer_len)
        # dataset = dataset.shuffle(seed=seed)

        def sharegpt_format(example, teacher_tokenizer, student_tokenizer):
            conversations = example['messages']
            message = []
            if isinstance(conversations, list):
                for conversation in conversations:
                    if isinstance(conversation, dict):
                        if conversation.get('role') == 'user':
                            message.append({"role": "user", "content": conversation.get('content', '')})
                        # elif conversation.get('role') == 'assistant':
                        #     message.append({"role": "assistant", "content": conversation.get('content', '')})
                        # elif conversation.get('from') == 'system':
                        #     message.insert(0, {"role": "system", "content": conversation.get('content', '')})
            # if not any(msg.get('role') == 'system' for msg in message):
            #     message.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            student_text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            teacher_text = teacher_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

            #delete     '<_begin_of_sentence_>'
            teacher_text = teacher_text[21:]

            answer = []
            if isinstance(conversations, list):
                for conversation in conversations:
                    if isinstance(conversation, dict):
                        if conversation.get('role') == 'assistant':
                            answer.append(conversation.get('content', ''))
                            break

            pure_answer = answer[0]
            if pure_answer.find("<think>\n")==0:
                pure_answer = pure_answer[8:]

            student_text = student_text + pure_answer
            teacher_text = teacher_text + pure_answer

            student_text += student_tokenizer.eos_token
            teacher_text += teacher_tokenizer.eos_token

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
            result = tokenizer(examples[column_name], truncation=True, max_length=max_seq_len,
                             padding="max_length")

            #for bug of non-stopping, we need to manully add eos in the case of truncation
            data_num = len(result.attention_mask)
            for i in range(data_num):
                if result.attention_mask[i][-1] == 1: #mostly be truncated
                    result.input_ids[i][-1] = tokenizer.eos_token_id
            return result

        teacher_tokenized_dataset = dataset.map(tokenize_function,
                                                fn_kwargs={"tokenizer": teacher_tokenizer, "column_name": "teacher_text"},
                                                batched=True,
                                                num_proc=14, remove_columns=["teacher_text"])
        teacher_tokenized_dataset = teacher_tokenized_dataset.rename_columns(
            {"input_ids": "teacher_input_ids", "attention_mask": "teacher_attention_mask"})


        #will not use student tokenizer
        student_tokenized_dataset = teacher_tokenized_dataset.map(tokenize_function, fn_kwargs={"tokenizer": student_tokenizer,
                                                                                                "column_name": "student_text"},
                                                                  batched=True,
                                                                  num_proc=14, remove_columns=["student_text"])
        student_tokenized_dataset = student_tokenized_dataset.rename_columns(
            {"input_ids": "student_input_ids", "attention_mask": "student_attention_mask"})

        print(f'finish read dataset')
        torch.save(student_tokenized_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname, weights_only=False)  # modifed 20250402 , weights_only=true



#only for lukedai/add_test_math dataset
def create_dataset_lukedai_test_add_math(local_rank,
                                    data_path,
                                    data_output_path,
                                    seed,
                                    tokenizer,
                                    max_seq_len,
                                    ):

    os.makedirs(data_output_path, exist_ok=True)
    fname = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")

    #depends on tokenizer, as need tokenzier text
    fname = f"only_train_full_{fname}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
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
        dataset = load_dataset(data_path[0], split="test") #here default to only 1 dataset




        # dataset = dataset.shuffle(seed=seed)

        def sharegpt_format(example, tokenizer):
            prompt = example['prompt']
            message = []
            message.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

            answer = []
            pure_answer = example['completion']

            if pure_answer.find("<think>")==0:
                pure_answer = pure_answer[7:]

            text = text + pure_answer

            text += tokenizer.eos_token


            return {"text": text}


        # Preprocess and tokenize the dataset

        map_kwargs = {}
        map_kwargs["num_proc"] = 52  # here is the parallel process number
        map_kwargs["desc"] = f"Applying chat template to {data_path[0]} dataset"

        print("Preprocessing and tokenizing dataset...")
        original_columns = dataset.column_names
        dataset = dataset.map(sharegpt_format,
                              fn_kwargs={"tokenizer": tokenizer},
                              remove_columns=original_columns, **map_kwargs)


        def tokenize_function(examples, tokenizer, column_name):
            result = tokenizer(examples[column_name], truncation=True, max_length=max_seq_len,
                             padding="max_length")

            #for bug of non-stopping, we need to manully add eos in the case of truncation
            data_num = len(result.attention_mask)
            for i in range(data_num):
                if result.attention_mask[i][-1] == 1: #mostly be truncated
                    result.input_ids[i][-1] = tokenizer.eos_token_id
            return result

        tokenized_dataset = dataset.map(tokenize_function,
                                                fn_kwargs={"tokenizer": tokenizer, "column_name": "text"},
                                                batched=True,
                                                num_proc=14, remove_columns=["text"])


        print(f'finish read dataset')
        torch.save(tokenized_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname, weights_only=False)  # modifed 20250402 , weights_only=true

