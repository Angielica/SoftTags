import json
import sys

import os

import torch

from models.models import EmbeddingNet
from models.trainer import Trainer
from utils.utility import create_folders, set_reproducibility
from dataset.dataloader import get_data

def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    init_seed = params['init_seed']

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["n_gpu"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["device"] = device

    create_folders(params)

    for i in range(params['n_runs']):
        dataset_name = params['dataset_name']
        params['seed'] = init_seed + i

        seed = params['seed']
        set_reproducibility(seed)

        exp = f'RUN_{i}_SEED_{seed}_DATASET_{dataset_name}'

        log_path = os.path.join(params['SAVE_FOLDER'], 'results', f'log_{exp}.txt')
        open(log_path, 'w').close()
        params['log_path'] = log_path

        best_path_model, last_path_model = f'best_model_{exp}.pt', f'last_model_{exp}.pt'
        params['best_path_model'] = os.path.join(params["SAVE_FOLDER"], 'models', best_path_model)
        params['last_path_model'] = os.path.join(params["SAVE_FOLDER"], 'models', last_path_model)

        path_training_loss, path_training_loss_log = f'training_loss_{exp}.pdf', f'training_log_loss_{exp}.pdf'
        params['path_training_loss'] = os.path.join(params["SAVE_FOLDER"], 'plots', path_training_loss)
        params['path_training_loss_log'] = os.path.join(params["SAVE_FOLDER"], 'plots', path_training_loss_log)

        # Create data
        train_loader = get_data(params)

        model = EmbeddingNet(embedding_dim=params['embedding_dim'])
        trainer = Trainer(params, model)

        if params['train']:
            trainer.train(train_loader)

        if params['test']:
            if params['load_best_model']:
                model.load_state_dict(torch.load(params['last_path_model']))
            else:
                model.load_state_dict(torch.load(params['best_path_model']))

            trainer.test(train_loader, model)


if __name__ == "__main__":
    main(sys.argv[1])