import numpy as np
import os
import random
import torch


def create_folders(params):
    if not os.path.exists(params["SAVE_FOLDER"]):
        os.mkdir(params["SAVE_FOLDER"])

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'models')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'models'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'plots')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'plots'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'results')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'results'))


def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False
