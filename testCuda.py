import torch
print(torch.version.cuda)
print("test done===========")

print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
torch.cuda.is_available()