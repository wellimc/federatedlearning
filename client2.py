import flwr as fl
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

import os
import yaml
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import preprocess2, inference, interpret

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 10
batch_size = 2
n_epochs = 50
n_epochs_stop = 10
label_name = 'Battery'

class TimeSeriesDataset(Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len

    def __getitem__(self, index):
        return self.X[index:index+self.seq_len], self.y[index+self.seq_len]


def main() -> None:
    # Model and data
    #model = mnist.LitAutoEncoder()
    #train_loader, val_loader, test_loader = mnist.load_data()

   
    class TSModel(nn.Module):
        def __init__(self, n_features, n_hidden=64, n_layers=2):
            super(TSModel, self).__init__()

            self.n_hidden = n_hidden
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=n_hidden,
                batch_first=True,
                num_layers=n_layers,
                dropout=0.5
            )
            self.linear = nn.Linear(n_hidden, 1)
            
        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            lstm_out = hidden[-1]  # output last hidden state output
            y_pred = self.linear(lstm_out)
            
            return y_pred

    # Flower client
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            #train(net, train_loader, epochs=1)
            t_loss = train_model(model,train_df)
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            accuracy = 80
            self.set_parameters(parameters)
            test_loss = test_model(model,test_df)

            return float(test_loss), len(test_loader), {"accuracy": float(accuracy)}



    file_name = "Trepn_2022.04.04_191744_q720_total.csv"
    data = preprocess2.load_data(file_name)

    train_df, test_df = preprocess2.prep_data(df=data, train_frac=0.6, plot_df=True)

    # create dataloaders
    train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # set up training
    n_features = train_df.shape[1]
    model = TSModel(n_features)

    criterion = torch.nn.MSELoss()  # L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('./model/model_b3.pt'))

    # Start client
    fl.client.start_numpy_client("[::]:8081", client=FlowerClient())

def train_model(model, train_df):
   
    print("Starting with model training...")

    # create dataloaders
    train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # set up training
    criterion = torch.nn.MSELoss()  # L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_hist = []

    # start training
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(1, n_epochs+1):
        running_loss = 0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            data = torch.Tensor(np.array(data))
            output = model(data)
            loss = criterion(output.flatten(), target.type_as(output))
            # if type(criterion) == torch.nn.modules.loss.MSELoss:
            #     loss = torch.sqrt(loss)  # RMSE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_hist.append(running_loss)
        print(f'Epoch {epoch} train loss: {round(running_loss,4)}')
       
    print("Training Completed.")

    return running_loss

def test_model( model,test_df ):

    print("Starting testting model ...")

    test_hist = []

    test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    criterion = torch.nn.MSELoss()  # L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # test loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.Tensor(np.array(data))
            output = model(data)
            loss = criterion(output.flatten(), target.type_as(output))
            test_loss += loss.item()
        test_loss /= len(test_loader)
        test_hist.append(test_loss)

    print(f'Test loss: {round(test_loss,4)}')

    hist = pd.DataFrame()
    hist['test_loss'] = test_hist

    print("Completed.")

    return test_loss

if __name__ == "__main__":
    main()
