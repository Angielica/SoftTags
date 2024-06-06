import json
import sys

import os

import torch

from models.models import SiameseNetwork
from models.trainer import Trainer
from utils.utility import create_folders, set_reproducibility
from utils.plotter import plot_roc_curve, plot_pr_curve
from dataset.dataloader import get_data

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score


def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    init_seed = params['init_seed']

    # if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # os.environ["CUDA_VISIBLE_DEVICES"] = params["n_gpu"]

    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    params["device"] = device

    print(torch.cuda.device_count())
    print(os.curdir)

    create_folders(params)

    for i in range(params['n_runs']):
        dataset_name = params['dataset_name']
        params['seed'] = init_seed + i

        seed = params['seed']
        angle = params['angle']
        set_reproducibility(seed)

        exp = f'RUN_{i}_SEED_{seed}_DATASET_{dataset_name}_angle_{angle}'

        log_path = os.path.join(params['SAVE_FOLDER'], 'results', f'log_{exp}.txt')
        open(log_path, 'w').close()
        params['log_path'] = log_path

        best_path_model, last_path_model = f'best_model_{exp}.pt', f'last_model_{exp}.pt'
        params['best_path_model'] = os.path.join(params["SAVE_FOLDER"], 'models', best_path_model)
        params['last_path_model'] = os.path.join(params["SAVE_FOLDER"], 'models', last_path_model)

        path_training_loss, path_training_loss_log = f'training_loss_{exp}.pdf', f'training_log_loss_{exp}.pdf'
        params['path_training_loss'] = os.path.join(params["SAVE_FOLDER"], 'plots', path_training_loss)
        params['path_training_loss_log'] = os.path.join(params["SAVE_FOLDER"], 'plots', path_training_loss_log)

        roc_plot, path_pr_curve = f'roc_{exp}.pdf', f'aupr_{exp}.pdf'
        params['path_roc'] = os.path.join(params["SAVE_FOLDER"], 'plots', roc_plot)
        params['path_pr_curve'] = os.path.join(params["SAVE_FOLDER"], 'plots', path_pr_curve)

        # Create data
        train_loader, test_loader = get_data(params)

        # model = EmbeddingNet(embedding_dim=params['embedding_dim'])
        model = SiameseNetwork(embedding_dim=params['embedding_dim'],
                               n_channels=params['n_channels'], h_size=params['img_size'],
                               w_size=params['img_size']).to(device)
        trainer = Trainer(params, model)

        if params['train']:
            trainer.train(train_loader)

        if params['test']:
            if params['load_best_model']:
                model.load_state_dict(torch.load(params['best_path_model']))
            else:
                model.load_state_dict(torch.load(params['last_path_model']))

            y_true, y_pred = trainer.test(test_loader, model)

            preds = np.around(y_pred, 0)
            confusion_matrix(y_true, preds)

            print(classification_report(y_true, preds))

            aucroc = roc_auc_score(y_true, y_pred)
            print('AUC: %.2f' % aucroc)

            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            plot_roc_curve(fpr, tpr, aucroc, params['path_roc'])

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc = auc(recall, precision)
            print(f'PR AUC: {auprc:.2f}')
            plot_pr_curve(precision, recall, params['path_pr_curve'])



if __name__ == "__main__":
    main(sys.argv[1])
