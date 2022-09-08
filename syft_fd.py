import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft as sy  # <-- NEW: import the Pysyft library
from torch.utils.data import Dataset, DataLoader
import preprocess
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 10
batch_size = 5
test_batch_size =  10
n_epochs = 50
n_epochs_stop = 10
label_name = 'Battery'
lr = 0.01
momentum = 0.5
no_cuda = False
log_interval = 10
save_model = True
seed = 1


hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice


with open("params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

data_dir = params['data_dir']
model_dir = params['model_dir']


class TimeSeriesDataset(Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len

    def __getitem__(self, index):
        return self.X[index:index+self.seq_len], self.y[index+self.seq_len]


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


def main() -> None:

     # determine the supported device
    def get_device():
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu') # don't have GPU 
        return device

    # convert a df to tensor to be used in pytorch
    def df_to_tensor(df):
        device = get_device()
        return torch.from_numpy(df.values).float().to(device)
   


    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    file_name = "Trepn_2022.04.04_183605_q360_total.csv"
    data = pd.read_csv(Path(data_dir, file_name))

    train_df, test_df = preprocess.prep_data(df=data, train_frac=0.6, plot_df=True)

    # create dataloaders
    train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # creating tensor from targets_df 
    df_tensor = df_to_tensor(train_df)
    #series_tensor = df_to_tensor(series)

    bob_train_dataset = sy.BaseDataset(df_tensor, df_tensor).send(bob)
    alice_train_dataset = sy.BaseDataset(df_tensor, df_tensor).send(alice)
    federated_train_dataset = sy.FederatedDataset([bob_train_dataset, alice_train_dataset])
    federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=False, batch_size=1)

    #federated_dataset = sy.FederatedDataset(train_df)
    #train_loader = sy.FederatedDataLoader(federated_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    #federated_train_loader = sy.FederatedDataLoader(federated_dataset.federate((bob, alice)), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
    #batch_size=args.batch_size, shuffle=True, **kwargs)

    #test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                   transforms.ToTensor(),
    #                   transforms.Normalize((0.1307,), (0.3081,))
    #               ])),
    #batch_size=args.test_batch_size, shuffle=False, **kwargs)

    n_features = train_df.shape[1]
    model = TSModel(n_features)

    criterion = torch.nn.MSELoss()  # L1Loss()
   # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #model.load_state_dict(torch.load('./model/model_b3.pt'))

    model = TSModel(n_features).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr) # TODO momentum is not supported at the moment

    for epoch in range(1, n_epochs + 1):
        train( model, device, federated_train_loader, optimizer, epoch)
        test( model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "test.pt")



def train( model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader): # <-- now it is a distributed dataset
        model.send(data.location) # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get() # <-- NEW: get the model back
        if batch_idx % log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader) * batch_size, #batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    main()
