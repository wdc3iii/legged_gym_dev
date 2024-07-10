import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import wandb
import os
import argparse
import numpy as np
import pickle
import json

def load_and_prepare_data(filename, tube_type):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    z = np.array(data['z'])
    v = np.array(data['v'])
    pz_x = np.array(data['pz_x'])
    pz_x_p1 = np.array(data['pz_x_p1'])

    if tube_type == 'rectangular':
        w_xy = pz_x - z
        w_xy_p1 = pz_x_p1 - z
    else:
        w = np.linalg.norm(pz_x - z, axis=1)
        w_p1 = np.linalg.norm(pz_x_p1 - z, axis=1)

    feature_data = np.hstack((z, v, w_xy if tube_type == 'rectangular' else w.reshape(-1, 1)))
    target_data = w_xy_p1 if tube_type == 'rectangular' else w_p1

    alphas = np.random.uniform(0, 1, size=(feature_data.shape[0], 1))
    feature_data = feature_data * alphas
    X = torch.tensor(feature_data, dtype=torch.float32)
    y = torch.tensor(target_data, dtype=torch.float32) if tube_type == 'rectangular' else torch.tensor(target_data.reshape(-1, 1), dtype=torch.float32)

    return X, y

def create_data_loaders(X, y, batch_size=64):
    dataset = TensorDataset(X, y[:, 1].unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

class TubeWidthPredictor(nn.Module):
    def __init__(self, input_size, num_units, num_layers, output_dim=1):
        super(TubeWidthPredictor, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, num_units), nn.ReLU()])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_units, num_units))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(num_units, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AsymmetricLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(AsymmetricLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, y_pred, y_true, tube_type, alpha):
        if tube_type == 'sphere':
            residual = y_true - y_pred
            loss = torch.where(residual <= 0, alpha * residual, (1 - alpha) * residual.abs())
            return self.huber(loss, torch.zeros_like(loss))
        elif tube_type == 'rectangular':
            y_true = y_true.squeeze(1)
            residual_x = y_true[:, 0] - y_pred[:, 0]
            residual_y = y_true[:, 1] - y_pred[:, 1]
            norm_residual = torch.sqrt(residual_x**2 + residual_y**2)
            loss = torch.where(norm_residual <= 0, alpha * norm_residual, (1 - alpha) * norm_residual.abs())
            return self.huber(loss, torch.zeros_like(loss))

def train_and_test(model, criterion, optimizer, train_loader, test_loader, tube_type, num_epochs=500, device='cpu'):
    best_test_loss = float('inf')
    model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0, leave=True):
        model.train()
        train_loss = 0
        for data, targets in tqdm(train_loader, desc="Training Batches", leave=False):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            alpha = data[:, 0]
            loss = criterion(outputs, targets, tube_type, alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        wandb.log({'Train Loss': train_loss / len(train_loader), 'Epoch': epoch})
        metrics = evaluate_model(model, test_loader, criterion, tube_type, device)
        for metric_name, metric_value in metrics.items():
            wandb.log({metric_name: metric_value, 'Epoch': epoch})
        
        if metrics['Test Loss (alpha=0.1)'] < best_test_loss:
            best_test_loss = metrics['Test Loss (alpha=0.1)']
            model_path = f"models/{tube_type}/model_epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            wandb.save(model_path)

def evaluate_model(model, test_loader, criterion, tube_type, device):
    model.eval()
    metrics = {}

    with torch.no_grad():
        for alpha in np.arange(0.1, 1.0, 0.1):
            test_loss = 0
            differences = []
            total_predictions = 0
            count_y_pred_gt_wt1 = 0

            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                alpha_tensor = torch.full((data.size(0), 1), alpha)
                data_with_alpha = torch.cat((alpha_tensor, data[:, 1:]), dim=1)

                outputs = model(data_with_alpha)
                loss = criterion(outputs, targets, tube_type, alpha)
                test_loss += loss.item()

                total_predictions += outputs.size(0)

                if tube_type == 'sphere':
                    greater_mask = outputs > targets
                    count_y_pred_gt_wt1 += greater_mask.sum().item()
                    less_mask = outputs < targets
                    differences.extend((targets[less_mask] - outputs[less_mask]).abs().tolist())

                elif tube_type == 'rectangular':
                    targets = targets.squeeze(1)
                    outputs_x_norm = outputs[:, 0]
                    outputs_y_norm = outputs[:, 1]
                    targets_x_norm = targets[:, 0]
                    targets_y_norm = targets[:, 1]

                    greater_mask_x = outputs_x_norm > targets_x_norm
                    greater_mask_y = outputs_y_norm > targets_y_norm
                    count_y_pred_gt_wt1 += (greater_mask_x | greater_mask_y).sum().item()

                    differences_x = (targets_x_norm - outputs_x_norm)[outputs_x_norm < targets_x_norm].abs().tolist()
                    differences_y = (targets_y_norm - outputs_y_norm)[outputs_y_norm < targets_y_norm].abs().tolist()
                    differences.extend(differences_x)
                    differences.extend(differences_y)

            avg_diff = sum(differences) / len(differences) if differences else 0
            test_loss /= len(test_loader)
            proportion_y_pred_gt_wt1 = count_y_pred_gt_wt1 / total_predictions if total_predictions > 0 else 0

            metrics[f'Test Loss (alpha={alpha:.1f})'] = test_loss
            metrics[f'Proportion y_pred > w_{{t+1}} (alpha={alpha:.1f})'] = proportion_y_pred_gt_wt1
            metrics[f'Avg Abs Diff y_pred < w_{{t+1}} (alpha={alpha:.1f})'] = avg_diff

    return metrics

def parse_list(arg):
    return [item.strip() for item in arg.split(',')]

def main(tube_type, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init()
    config = wandb.config

    X, y = load_and_prepare_data(filename, tube_type)
    train_loader, test_loader = create_data_loaders(X, y, batch_size=64)

    input_size = 7
    output_dim = 2 if tube_type == 'rectangular' else 1

    model = TubeWidthPredictor(input_size=input_size, num_units=config.num_units, num_layers=config.num_layers, output_dim=output_dim)
    criterion = AsymmetricLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_and_test(model, criterion, optimizer, train_loader, test_loader, tube_type, num_epochs=config.num_epoch, device=device)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the wandb sweep with multiple configurations.")
    parser.add_argument("--tube_type", type=parse_list, default="rectangular", help="List of tube types to process.")
    parser.add_argument("--run_name", type=str, default="fxqv2jkf", help="Folder name where run data is stored.")

    args = parser.parse_args()

    wandb.login(key="70954bb73c536b7f5b23ef315c7c19b511e8a406")

    with open('sweep_config.json', 'r') as file:
        all_configs = json.load(file)
        config = all_configs['tube']

    for tube_type in args.tube_type:
        config['parameters']['tube_type']['value'] = tube_type
        project_name = f"{args.config_key}_experiment_{tube_type}"
        sweep_id = wandb.sweep(config, project=project_name)
        wandb.agent(sweep_id, lambda: main(tube_type, f"rom_tracking_data/{args.run_name}/dataset.pickle"))