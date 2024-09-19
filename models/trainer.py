import numpy as np
from sklearn.model_selection import KFold

from time import time
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from models.losses import TripletLoss, TripletClassifierLoss
from models.models import SiameseNetwork
from utils.plotter import plot_loss_training, plot_loss_training_log, plot_loss, plot_loss_log
from utils.utility import reset_weights, EarlyStopper


class Trainer:
    def __init__(self, params, model):
        self.params = params
        self.device = params['device']

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        # self.criterion = TripletLoss(margin=params['margin'], p=params['p'], eps=params['eps'],
        #                              reduction=params['reduction'])

        self.criterion = TripletClassifierLoss()

        self.tensorboard_path = self.params['tensorboard_path']
        self.writer = SummaryWriter(self.tensorboard_path)

    def train_step(self, a, p, n):
        self.optimizer.zero_grad()

        # d_aa, d_ap, d_an, d_pn = self.model(a, p, n)
        d_ap, d_an = self.model(a, p, n)
        loss = self.criterion(d_ap, d_an)
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, val_loader):
        self.model.eval()

        val_losses = []

        with torch.no_grad():
            for a, p, n, _ in tqdm(val_loader, total=len(val_loader)):
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                d_ap, d_an = self.model(a, p, n)

                loss = self.criterion(d_ap, d_an)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def test(self, test_loader, model):
        model.eval()

        with torch.no_grad():
            count = 0
            # for x1, x2, y in tqdm(test_loader):
            for a, p, yap, n, yan in tqdm(test_loader, total=len(test_loader)):
                # x1, x2 = x1.to(self.device), x2.to(self.device)
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                dap = model.inference(a, p)
                dan = model.inference(a, n)

                if count == 0:
                    y_true = yap
                    y_pred = dap.detach().cpu()

                    y_true = torch.cat((y_true, yan))
                    y_pred = torch.cat((y_pred, dan.detach().cpu()))

                else:
                    y_true = torch.cat((y_true, yap), dim=0)
                    y_pred = torch.cat((y_pred, dap.detach().cpu()), dim=0)

                    y_true = torch.cat((y_true, yan), dim=0)
                    y_pred = torch.cat((y_pred, dan.detach().cpu()), dim=0)

                count += 1
        return y_true, y_pred

    def train(self, train_loader, val_loader=None, early_stop=True):
        train_losses, val_losses = [], []
        best_loss = np.Inf

        n_epochs = self.params['n_epochs']

        time_start = time()
        print(f'Starting training phase. Starting time: {time_start} \n')
        print(f"Number of steps: {len(train_loader)}")

        early_stopped = False
        early_stopper = EarlyStopper(patience=10, min_delta=0.001)

        with open(self.params['log_path'], 'a') as f:
            f.write(f'Starting training phase. Starting time: {time_start} \n')

        try:
            for epoch in range(self.params['n_epochs']):
                self.model.train()
                epoch_loss = []

                for a, p, n, _ in tqdm(train_loader, total=len(train_loader)):
                    a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                    loss = self.train_step(a, p, n)
                    epoch_loss.append(loss.item())

                epoch_loss = np.mean(epoch_loss)
                train_losses.append(epoch_loss)

                print(f'Epoch: {epoch}/{n_epochs}, Train loss: {epoch_loss}')
                with open(self.params['log_path'], 'a') as f:
                    f.write(f'Epoch: {epoch} / {n_epochs}, Train loss: {epoch_loss} \n')

                self.writer.add_scalar('train_loss', epoch_loss, epoch)

                if val_loader is not None:
                    val_loss = self.evaluate(val_loader)
                    val_losses.append(val_loss)
                    current_loss = val_loss

                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(self.model.state_dict(), self.params["best_path_model"])

                    print(f'Epoch: {epoch} / {n_epochs}, Val loss: {val_loss}')
                    with open(self.params['log_path'], 'a') as f:
                        f.write(f'Epoch: {epoch} / {n_epochs}, Val loss: {val_loss} \n')
                    self.writer.add_scalar('val_loss', val_loss, epoch)

                else:
                    current_loss = epoch_loss
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save(self.model.state_dict(), self.params["best_path_model"])

                if early_stop and early_stopper.early_stop(current_loss):
                    early_stopped = True
                    break

            if early_stopped:
                print('Early stopping \n')

                with open(self.params['log_path'], 'a') as filehandle:
                    filehandle.write(f'Early stopping \n')

        except KeyboardInterrupt:
            print('Early training phase ..')

        end_train = time()
        print(f'End training phase. Elapsed time: {end_train - time_start} \n')

        with open(self.params['log_path'], 'a') as f:
            f.write(f'End training phase. Elapsed time: {end_train - time_start} \n')

        torch.save(self.model.state_dict(), self.params["last_path_model"])

        if val_loader is None:
            plot_loss_training(train_losses, self.params["path_training_loss"] + '.pdf')
            plot_loss_training_log(train_losses, self.params["path_training_loss_log"] + '.pdf')
        else:
            plot_loss(train_losses, val_losses, self.params['path_training_loss'] + '.pdf')
            plot_loss_log(train_losses, val_losses, self.params['path_training_loss_log'] + '.pdf')


class TrainerCrossValidation:
    def __init__(self, params):
        self.params = params
        self.device = params['device']

        self.criterion = TripletClassifierLoss()

    def test(self, test_loader, model):
        model.eval()

        with torch.no_grad():
            count = 0
            # for x1, x2, y in tqdm(test_loader):
            for a, p, yap, n, yan in tqdm(test_loader):
                # x1, x2 = x1.to(self.device), x2.to(self.device)
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                dap = model.inference(a, p)
                dan = model.inference(a, n)

                if count == 0:
                    y_true = yap
                    y_pred = dap.detach().cpu()

                    y_true = torch.cat((y_true, yan))
                    y_pred = torch.cat((y_pred, dan.detach().cpu()))

                else:
                    y_true = torch.cat((y_true, yap), dim=0)
                    y_pred = torch.cat((y_pred, dap.detach().cpu()), dim=0)

                    y_true = torch.cat((y_true, yan), dim=0)
                    y_pred = torch.cat((y_pred, dan.detach().cpu()), dim=0)

                count += 1
        return y_true, y_pred

    def train(self, train_dataset):
        kfold = KFold(n_splits=self.params['k_fold'], shuffle=True)
        best_loss = np.Inf

        n_epochs = self.params['n_epochs']

        time_start = time()
        print(f'Starting training phase. Starting time: {time_start} \n')

        try:
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                with open(self.params['log_path'], 'a') as f:
                    f.write(f'FOLD: {fold} \n')

                train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['batch_size'],
                                                           sampler=train_subsampler)
                val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['batch_size'],
                                                         sampler=val_subsampler)

                # Init the neural network
                model = SiameseNetwork(embedding_dim=self.params['embedding_dim'],
                               n_channels=self.params['n_channels'], h_size=self.params['img_size'],
                               w_size=self.params['img_size']).to(self.device)
                model.apply(reset_weights)

                self._train_on_fold(fold, model, train_loader, n_epochs)
                valid_loss = self._evaluate_on_fold(fold, model, val_loader)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), self.params["best_path_model"])
        except KeyboardInterrupt:
            print('Early training phase ..')

        end_train = time()
        print(f'End training phase. Elapsed time: {end_train - time_start} \n')

        with open(self.params['log_path'], 'a') as f:
            f.write(f'End training phase. Elapsed time: {end_train - time_start} \n')

    def _train_on_fold(self, fold, model, train_loader, n_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params['lr'])

        train_losses = []

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = []

            for a, p, n, _ in tqdm(train_loader, total=len(train_loader)):
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                optimizer.zero_grad()
                d_ap, d_an = model(a, p, n)
                loss = self.criterion(d_ap, d_an)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            epoch_loss = np.mean(epoch_loss)
            train_losses.append(epoch_loss)

            print(f'Fold: {fold} --> Epoch: {epoch}/{n_epochs}, Train loss: {epoch_loss}')
            with open(self.params['log_path'], 'a') as f:
                f.write(f'Fold: {fold} --> Epoch: {epoch} / {n_epochs}, Train loss: {epoch_loss}')

        plot_loss_training(train_losses, self.params["path_training_loss"] + f'_FOLD_{fold}.pdf')
        plot_loss_training_log(train_losses, self.params["path_training_loss_log"] + f'_FOLD_{fold}.pdf')

        return train_losses

    def _evaluate_on_fold(self, fold, model, val_loader):
        model.eval()

        val_losses = []

        with torch.no_grad():
            for a, p, n, _ in val_loader:
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                d_ap, d_an = model(a, p, n)

                loss = self.criterion(d_ap, d_an)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        print(f'FOLD: {fold}, Val loss: {val_loss}')
        with open(self.params['log_path'], 'a') as f:
            f.write(f'FOLD: {fold}, Val loss: {val_loss}')

        return val_loss

