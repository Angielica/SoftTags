import numpy as np
from time import time
import torch
from tqdm import tqdm

from models.losses import TripletLoss, TripletClassifierLoss
from utils.plotter import plot_loss_training, plot_loss_training_log, plot_loss, plot_loss_log


class Trainer:
    def __init__(self, params, model):
        self.params = params
        self.device = params['device']

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        # self.criterion = TripletLoss(margin=params['margin'], p=params['p'], eps=params['eps'],
        #                              reduction=params['reduction'])

        self.criterion = TripletClassifierLoss()

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
            for a, p, n, _ in val_loader:
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

    def train(self, train_loader, val_loader=None):
        train_losses, val_losses = [], []
        best_loss = np.Inf

        n_epochs = self.params['n_epochs']

        time_start = time()
        print(f'Starting training phase. Starting time: {time_start} \n')
        print(f"Number of steps: {len(train_loader)}")

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
                    f.write(f'Epoch: {epoch} / {n_epochs}, Train loss: {epoch_loss}')

                if val_loader is not None:
                    val_loss = self.evaluate(val_loader)
                    val_losses.append(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(self.model.state_dict(), self.params["best_path_model"])

                    print(f'Epoch: {epoch} / {n_epochs}, Val loss: {val_loss}')
                    with open(self.params['log_path'], 'a') as f:
                        f.write(f'Epoch: {epoch} / {n_epochs}, Val loss: {val_loss}')

                else:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save(self.model.state_dict(), self.params["best_path_model"])

        except KeyboardInterrupt:
            print('Early training phase ..')

        end_train = time()
        print(f'End training phase. Elapsed time: {end_train - time_start} \n')

        with open(self.params['log_path'], 'a') as f:
            f.write(f'End training phase. Elapsed time: {end_train - time_start} \n')

        torch.save(self.model.state_dict(), self.params["last_path_model"])

        if val_loader is None:
            plot_loss_training(train_losses, self.params["path_training_loss"])
            plot_loss_training_log(train_losses, self.params["path_training_loss_log"])
        else:
            plot_loss(train_losses, val_losses, self.params['path_training_loss'])
            plot_loss_log(train_losses, val_losses, self.params['path_training_loss_log'])
