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


def reset_weights(m):
  """
    Try resetting model weights to avoid
    weight leakage.
  """
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                return True

        return False
