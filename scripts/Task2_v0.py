# import dependencies
import torch
import sklearn
import datasets
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
# from your_model_file import YourModelClass
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn as nn

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

ext_data = pd.read_csv(r"D:/Academics/W24/Neural Networks Theory And Implementation/Project Files/tasks/External-Dataset_for_Task2.csv")

########################################################
# Entry point
########################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True).to(device)

class MoLFormerWithRegressionHead(nn.Module):
    def __init__(self, base_model, hidden_dim=768):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(pooled_output)

regression_model = MoLFormerWithRegressionHead(model).to(device)
regression_model.load_state_dict(torch.load("molformer_regression_model.pth"))
regression_model.eval()

class SMILESDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self.prepare_data()

    def prepare_data(self):
        processed = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            encoded_input = self.tokenizer(
                row["SMILES"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            target_tensor = torch.tensor(row["label"], dtype=torch.float)
            processed.append({
                "input_ids": encoded_input["input_ids"].squeeze(0),
                "attention_mask": encoded_input["attention_mask"].squeeze(0),
                "target": target_tensor
            })
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def compute_gradients(model, data_loader, loss_fn):
    gradients = []
    model.eval()
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)
        
        model.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        grad = [param.grad.clone().detach().cpu() for param in model.parameters()]
        gradients.append(grad)
    
    return gradients

def lissa_approximation(model, test_loader, loss_fn, iter=100, damping=0.01):
    hvp_estimate = [torch.zeros_like(p, device=device) for p in model.parameters()]
    for _ in range(iter):
        model.zero_grad()
        test_batch = next(iter(test_loader))
        input_ids = test_batch["input_ids"].to(device)
        attention_mask = test_batch["attention_mask"].to(device)
        targets = test_batch["target"].to(device).unsqueeze(1)
        
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        grad_test = [p.grad.clone() for p in model.parameters()]
        
        hvp_estimate = [
            damping * hvp + g_t 
            for hvp, g_t in zip(hvp_estimate, grad_test)
        ]
    
    return hvp_estimate


if __name__ == "__main__":
    ext_dataset = SMILESDataset(ext_data, tokenizer)
    ext_loader = DataLoader(ext_dataset, batch_size=16, shuffle=False)

    loss_fn = nn.MSELoss()
    test_loader = DataLoader(ext_dataset, batch_size=1, shuffle=True)
    hvp = lissa_approximation(regression_model, test_loader, loss_fn)

    def compute_influence_scores(model, ext_loader, hvp):
        influence_scores = []
        gradients = compute_gradients(model, ext_loader, loss_fn)
        for grad in gradients:
            score = sum([(g * h).sum().item() for g, h in zip(grad, hvp)])
            influence_scores.append(score)
        return influence_scores

    influence_scores = compute_influence_scores(regression_model, ext_loader, hvp)

    top_k = 500
    ext_data["influence_score"] = influence_scores
    selected_data = ext_data.nlargest(top_k, "influence_score")

    selected_data.to_csv("selected_external_data.csv", index=False)
