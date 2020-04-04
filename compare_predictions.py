"""
Script that verifies that the predictions ran in PyTorch, Python's ONNX runtime and in the ONNX C++ backend are the same.
"""
import onnxruntime
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch
import requests
import csv


FP_SIZE = 2048
RADIUS = 2
BATCH_SIZE = 20
N_EPOCHS = 40


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def calc_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, RADIUS, nBits=FP_SIZE)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a

# load the SMILES
with open('training/CHEMBL1829.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    smiles = [x[1] for x in reader]

# calc the FPs
descs = [calc_morgan_fp(smi) for smi in smiles]

# run predictions with the Python ONNX runtime
ort_session = onnxruntime.InferenceSession("src/mlp.onnx")
ort_preds = np.array([ort_session.run(None, {ort_session.get_inputs()[0].name: fps})[0] for fps in descs])

# run the redictions with the C++ REST backend
def pred_cxx(smiles):
    res = requests.post('http://localhost:9999/predict', data=smiles)
    return res.json()['pred']

cxx_preds = np.array([pred_cxx(smi) for smi in smiles])

# compare C++ REST backend and Python ONNX runtime predictions
np.testing.assert_allclose(cxx_preds, ort_preds, rtol=1e-03, atol=1e-05)

# run the predictions with the PyTorch model
tsm = torch.jit.load("src/mlp.pt")
tsm.eval()
with torch.no_grad():
    pt_preds = tsm(torch.Tensor(descs))

# compare C++ REST backend and Python ONNX runtime predictions
np.testing.assert_allclose(to_numpy(pt_preds), cxx_preds, rtol=1e-03, atol=1e-05)
