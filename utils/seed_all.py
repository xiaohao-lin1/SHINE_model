import random

import numpy as np
import torch


#hello
def seed_all(torch_deterministic,seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('cuda is available')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    seed_all()
    # Get the current random seed for torch
    current_seed = torch.initial_seed()
    # Print the current seed
    print("Current PyTorch random seed:", current_seed)

