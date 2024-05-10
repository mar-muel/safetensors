import torch
from safetensors import safe_open, safe_write
from safetensors.torch import save_file, save_tensors
import sys


tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}
save_file(tensors, "model.safetensors")

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        print(f.get_tensor(key))

new_weight = torch.ones((1024, 1024))
overwrite_key = 'weight1'

save_tensors({overwrite_key: new_weight}, 'model.safetensors')

with safe_open("model.safetensors", framework="torch", device="cpu") as f:
    nw = f.get_tensor(overwrite_key)
print(nw)
