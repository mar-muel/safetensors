import numpy as np
from safetensors import safe_open, safe_write
from safetensors.numpy import save_file, save_tensors
import sys

def _is_little_endian(tensor: np.ndarray) -> bool:
    byteorder = tensor.dtype.byteorder
    if byteorder == "=":
        if sys.byteorder == "little":
            return True
        else:
            return False
    elif byteorder == "|":
        return True
    elif byteorder == "<":
        return True
    elif byteorder == ">":
        return False
    raise ValueError(f"Unexpected byte order {byteorder}")


def _tobytes(tensor: np.ndarray) -> bytes:
    if not _is_little_endian(tensor):
        tensor = tensor.byteswap(inplace=False)
    return tensor.tobytes()


def to_bytes(np_array: np.ndarray):
    return {"dtype": np_array.dtype.name, "shape": np_array.shape, "data": _tobytes(np_array)}


tensors = {
   "weight1": np.zeros((1024, 1024)),
   "weight2": np.zeros((1024, 1024))
}
save_file(tensors, "model.safetensors")

with safe_open("model.safetensors", framework="np", device="cpu") as f:
    for key in f.keys():
        print(f.get_tensor(key))

new_weight = np.ones((1024, 1024))
overwrite_key = 'weight1'

# new_weight = to_bytes(new_weight)
# with safe_write("model.safetensors", framework="np", device="cpu") as f:
#     f.set_tensor(overwrite_key, new_weight)

save_tensors({overwrite_key: new_weight}, 'model.safetensors')


with safe_open("model.safetensors", framework="np", device="cpu") as f:
    nw = f.get_tensor(overwrite_key)
print(nw)
