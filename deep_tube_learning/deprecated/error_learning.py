import numpy as np
import os
import pickle
from pathlib import Path
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse
import json

def load_and_process_data(run_id):
    file_path = f"rom_tracking_data/{run_id}/dataset.pickle"

    # Load data from the pickle file
    with open(file_path, "rb") as f:
        epoch_data = pickle.load(f)

    processed_data = {}
    for key in epoch_data:
        processed_data[key] = epoch_data[key][:-1, :]

    processed_data['e'] = processed_data['pz_x'] - processed_data['z']
    processed_data['e_p1'] = np.roll(processed_data['e'], -1, axis=0)
    processed_data['e_p1'][-1, :] = 0 # for wrap around...

    return processed_data

def save_data(run_id, data):
    output_path = Path('../rom_tracking_data', run_id, "processed_data_with_errors.pickle")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

def convert_to_tensor_input(data):
    features = torch.tensor(data['e'], dtype=torch.float32)
    targets = torch.tensor(data['e_p1'], dtype=torch.float32)
    return features, targets

def create_data_loaders(X, y, batch_size=16384):
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

class ErrorDynamicsModel(nn.Module):
    def __init__(self, input_dim, num_units, num_layers):
        super(ErrorDynamicsModel, self).__init__()
        layers = [nn.Linear(input_dim, num_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(num_units, num_units), nn.ReLU()])
        layers.append(nn.Linear(num_units, input_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CustomErrorDynamicsLoss(nn.Module):
    def __init__(self):
        super(CustomErrorDynamicsLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.norm(y_pred - y_true, p=2, dim=1).mean()

def train_and_test(model, criterion, optimizer, train_loader, test_loader, device, num_epochs=50):
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        total_train_loss = 0
        ii = 0
        for data, targets in train_loader:
            print("\t", ii)
            ii += 1
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

        wandb.log({'Epoch': epoch, 'Train Loss': total_train_loss / len(train_loader), 'Test Loss': total_test_loss / len(test_loader)})
        # print({'Epoch': epoch, 'Train Loss': total_train_loss / len(train_loader), 'Test Loss': total_test_loss / len(test_loader)})

        if epoch == 0 or total_test_loss < best_test_loss:
            best_test_loss = total_test_loss
            model_path = f"models/model_epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

def main(run_id):
    wandb.init()
    config = wandb.config
    
    data = load_and_process_data(run_id)
    X, y = convert_to_tensor_input(data)
    train_loader, test_loader = create_data_loaders(X, y)

    input_dim = X.size(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ErrorDynamicsModel(input_dim, config.num_units, config.num_layers).to(device)
    criterion = CustomErrorDynamicsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_and_test(model, criterion, optimizer, train_loader, test_loader, device, config.num_epochs)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the wandb sweep with multiple configurations.")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID for the data.")
    args = parser.parse_args()

    wandb.login(key="70954bb73c536b7f5b23ef315c7c19b511e8a406")

    with open('sweep_config.json', 'r') as file:
        all_configs = json.load(file)
        sweep_config = all_configs.get('error', all_configs['default'])

    project_name = f"model_training_sweep_{args.config_key}"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, lambda: main(args.run_id))