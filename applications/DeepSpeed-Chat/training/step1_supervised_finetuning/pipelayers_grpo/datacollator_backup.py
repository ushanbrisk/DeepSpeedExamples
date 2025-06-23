import torch
import os

class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, samples):
        print(f" process: {os.getpid()}")
        input_ids_list, labels_list = [], []

        bz = len(samples)
        for idx in range(bz):
            if len(samples[idx]["input_ids"]) == self.max_len:
                break
            else:
                samples[idx]["input_ids"] = samples[idx]["input_ids"] + \
                    [self.tokenizer.pad_token_id] * (self.max_len -len(samples[idx]["input_ids"]))


        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, samples, return_tensors="pt",max_length=self.max_len
        )

        # for instance in batch:
        #     input_ids = torch.tensor(instance["input_ids"], dtype=torch.int64)
        #     # input_ids = instance["input_ids"]
        #     input_ids_list.append(input_ids)
        #     # labels_list.append(instance["labels"])
        #     labels = input_ids.clone()  #copy from DataCollatorForLanguageModeling
        #     if self.tokenizer.pad_token_id is not None:
        #         labels[labels==self.tokenizer.pad_token_id] = -100
        #     labels_list.append(labels)

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))
        return ((batch['input_ids'], batch['labels']), batch['labels'])

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded





