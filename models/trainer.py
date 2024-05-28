import numpy as np

from losses import TripletLoss

from time import time
import torch

from utils.plotter import plot_loss_training, plot_loss_training_log
class Trainer:
    def __init__(self, params, model):
        self.params = params
        self.device = params['device']

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        self.criterion = TripletLoss(margin=params['margin'], p=params['p'], eps=params['eps'],
                                     reduction=params['reduction'])


    def train_step(self, a, p, n):
        self.optimizer.zero_grad()
        a_emb = self.model(a)
        p_emb = self.model(p)
        n_emb = self.model(n)

        loss = self.criterion(a_emb, p_emb, n_emb)
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, val_loader):
        self.model.eval()

        val_losses = []

        with torch.no_grad():
            for a, p, n, _ in val_loader:
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                a_emb = self.model(a)
                p_emb = self.model(p)
                n_emb = self.model(n)

                loss = self.criterion(a_emb, p_emb, n_emb)
                val_losses.append(loss.item())

        return np.mean(val_losses)


    def test(self, test_loader, model):
        model.eval()

        val_losses = []

        with torch.no_grad():
            for x1, x2, _ in test_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)

                x1_emb = self.model(x1)
                x2_emb = self.model(x2)

                # TODO


        return None


    def train(self, train_loader, val_loader=None):
        train_losses, val_losses = [], []
        best_loss = np.Inf

        n_epochs = self.params['n_epochs']

        time_start = time()

        with open(self.params['log_path'], 'w') as f:
            f.write(f'Starting training phase. Starting time: {time_start} \n')

        try:
            for epoch in range(self.params['n_epochs']):
                self.model.train()
                epoch_loss = []

                for a, p, n, _ in train_loader:
                    a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                    loss = self.train_step(a, p, n)
                    epoch_loss.append(loss.item())

                epoch_loss = np.mean(epoch_loss)
                train_losses.append(epoch_loss)

                print(f'Epoch: {epoch/n_epochs}, Train loss: {epoch_loss}')
                with open(self.params['log_path'], 'w') as f:
                    f.write(f'Epoch: {epoch / n_epochs}, Train loss: {epoch_loss}')

                if val_loader is not None:
                    val_loss = self.evaluate(val_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(self.model.state_dict(), self.params["best_path_model"])

                    print(f'Epoch: {epoch / n_epochs}, Val loss: {val_loss}')
                    with open(self.params['log_path'], 'w') as f:
                        f.write(f'Epoch: {epoch / n_epochs}, Val loss: {val_loss}')


        except KeyboardInterrupt:
            print('Early training phase ..')

        end_train = time()
        print(f'End training phase. Elapsed time: {end_train - time_start} \n')

        with open(self.params['log_path'], 'a') as f:
            f.write(f'End training phase. Elapsed time: {end_train - time_start} \n')

        torch.save(self.model.state_dict(), self.params["last_path_model"])

        plot_loss_training(train_losses, self.params["path_training_loss"])
        plot_loss_training_log(train_losses, self.params["path_training_loss_log"])




