import torch

status = torch.cuda.is_available()
print(status)