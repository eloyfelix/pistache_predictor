"""
This script trains a multi layer perceptron model and serializes it to be used in LibTorch (C++).
No test nor validation is done as we only need a "dummy" model to be exported for the demo.
"""
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


FP_SIZE = 2048
RADIUS = 2
BATCH_SIZE = 20
N_EPOCHS = 40


def calc_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, RADIUS, nBits=FP_SIZE)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a


class DatasetSMILES(Dataset):

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.transform = transforms.Compose([calc_morgan_fp, torch.Tensor])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fp = self.data.iloc[index, 1]
        labels = self.data.iloc[index, 0]
        fp = self.transform(fp)

        return fp, labels


train_dataset = DatasetSMILES('CHEMBL1829.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)


# multi layer perceptron model definition
class MLP(torch.nn.Module):

    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        out = torch.sigmoid(self.fc4(h3))
        return out


mlp = MLP(FP_SIZE)
# loss function and optimizer definition
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.05)

# model training
for epoch in range(N_EPOCHS):
    for i, (fps, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = mlp(fps)

        # calc the loss
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, N_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE, loss.item()))


# serialize the model to be loaded in LibTorch (C++)
tsm = torch.jit.trace(mlp, torch.ones(FP_SIZE))
tsm.save("mlp.pt")
