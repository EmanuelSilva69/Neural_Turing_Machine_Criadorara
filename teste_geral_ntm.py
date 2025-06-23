import torch
print(torch.__version__)
print(torch.cuda.is_available())  # should return True
print(torch.cuda.get_device_name(0))  # name of your GPU