import torch
from typing import List, Union, Any
def hash_tensor(tensor: Union[torch.Tensor, Any] ):
    if isinstance(tensor, torch.Tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))
    elif isinstance(tensor, (tuple,list)):
        result = []
        for x in tensor:
            result.extend(x.reshape(-1).tolist())
        result = tuple(result)
        return hash(result)
