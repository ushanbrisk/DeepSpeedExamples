import torch
def print_mem(rank, device, info='' ):
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated()  # allocated mem
    max_allocated = torch.cuda.max_memory_allocated()  # maximum allocated mem in history
    reserved = torch.cuda.memory_reserved()  # allocated + cache mem
    max_reserved = torch.cuda.max_memory_reserved()  # maximum allocated + cache mem in history


    print(f"rank:{rank},device:{device}, {info} ) - \
        Total memory: {total / (1024**2)} MB, \
        Allocated: {allocated / 1024**2:.2f} MB, \
        Max Allocated: {max_allocated / 1024**2:.2f} MB, \
        Reserved: {reserved / 1024**2:.2f} MB, \
        Max Reserved: {max_reserved / 1024**2:.2f} MB, \
        device name {torch.cuda.get_device_name(device)}"
    )
