#collect multiple data
#make them of the same length, then iterator output

import torch
from trl.trainer.utils import pad


class BatchDataBuffer():

    def __init__(self, max_lenght=10000):
        self.max_len = max_lenght
        self.buffer = []  #store input unprocessed data
        self.new_data_buffer = [] #store processed data
        self.data_num = 0

    def add(self, data):
        self.buffer.append(data)  #storing [batch1, batch2, batch3,...batch_gas]

    def adjust(self, padding_value=0):
        data_num = len(self.buffer)
        self.data_num = data_num

        new_prompt_completion_id = []
        prompt_completion_id = [data[0][0] for data in self.buffer]
        length_before_padding = [data.shape[1] for data in prompt_completion_id]
        #pad each batch data, so total gas data has  the same length
        prompt_completion_id = pad(prompt_completion_id, padding_value=padding_value)
        max_length = prompt_completion_id.shape[2]
        for i in range(prompt_completion_id.shape[0]):
            new_prompt_completion_id.append(prompt_completion_id[i])

        #here prompt_completion_id shape[gas, batch_size, max_length]

        new_attention_mask = []
        attention_mask =  [data[0][1] for data in self.buffer]
        attention_mask = pad(attention_mask, padding_value=0)
        for i in range(attention_mask.shape[0]):
            new_attention_mask.append(attention_mask[i])

        #logits_to_keep is of the same for the same batch
        #will include padding part of answer, but it will leave to attention_mask to decide
        logits_to_keep = [int(data[1][1]) for data in self.buffer]
        new_logits_to_keep = []
        for n_len, n_logits in zip(length_before_padding, logits_to_keep):
            n_logits += max_length - n_len
            new_logits_to_keep.append(torch.tensor(n_logits))

        advantage = [data[1][2] for data in self.buffer]

        for i in range(data_num):
            send_to_next_stage =  (new_prompt_completion_id[i], new_attention_mask[i])
            send_to_last_stage = (new_prompt_completion_id[i], new_logits_to_keep[i], advantage[i])
            batch = (send_to_next_stage, send_to_last_stage)
            self.new_data_buffer.append(batch)

    def steps(self):

        #since extend length, we should shrink in loss function to save computation
        for i in range(self.data_num):
            yield self.new_data_buffer[i]

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)




