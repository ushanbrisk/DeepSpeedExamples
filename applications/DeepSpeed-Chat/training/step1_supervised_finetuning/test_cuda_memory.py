import torch
 
# 模型初始化
linear1 = torch.nn.Linear(1024,1024, bias=False).cuda() # + 4194304
print(torch.cuda.memory_allocated())
linear2 = torch.nn.Linear(1024, 1, bias=False).cuda() # + 4096
print(torch.cuda.memory_allocated())
 
# 输入定义
inputs = torch.tensor([[1.0]*1024]*1024).cuda() # shape = (1024,1024) # + 4194304
print(torch.cuda.memory_allocated())
 
# 前向传播
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304 + 512
print(torch.cuda.memory_allocated())
 
# 后向传播
loss.backward() # memory - 4194304 + 4194304 + 4096
print(torch.cuda.memory_allocated())
 
# 再来一次~
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304  (512没了，因为loss的ref还在)
print(torch.cuda.memory_allocated())
loss.backward() # memory - 4194304
print(torch.cuda.memory_allocated())
