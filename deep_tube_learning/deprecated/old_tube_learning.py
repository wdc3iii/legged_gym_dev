import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import ast
from tqdm import tqdm
import wandb
import os
import argparse

def safe_eval(col):
    try:
        return ast.literal_eval(col)
    except ValueError:
        return col  # Return as is if it's not a string representation of a list

def convert_to_tensor_input(row):
    flat_list = []
    for item in row:
        if isinstance(item, list):
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list

def load_and_prepare_data(filename, tube_type):
    df = pd.read_csv(filename)
    list_columns = ['u_t', 'z_t', 'v_t'] + (['w_xy_t', 'w_xy_{t+1}'] if tube_type == 'rectangular' else [])
    feature_columns = ['w_t', 'z_t', 'v_t']
    
    for col in list_columns:
        df[col] = df[col].apply(safe_eval)

    X = torch.tensor(df[feature_columns].apply(convert_to_tensor_input, axis=1).tolist(), dtype=torch.float32)
    
    if tube_type == 'rectangular':
        target_columns = ['w_xy_t', 'w_xy_{t+1}']
    else:
        # target_columns = ['w_t', 'w_{t+1}']
        target_columns = ['w_{t+1}']

    # Construct the target tensor manually from rows
    y_data = df[target_columns].apply(lambda row: [row[col] for col in target_columns], axis=1).tolist()
    y = torch.tensor(y_data, dtype=torch.float32)
    
    return X, y

def create_data_loaders(X, y, batch_size=64):
    # Create a TensorDataset
    dataset = TensorDataset(X, y[:, 1].unsqueeze(1))

    # Split data into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader objects
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
        self.layers.append(nn.Linear(num_units, output_dim))  # Dynamically set output dimension

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AsymmetricLoss(nn.Module):
    def __init__(self, alpha=0.9, delta=1.0):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, y_pred, y_true, tube_type):
        if tube_type == 'sphere':
            residual = y_true - y_pred
            loss = torch.where(residual <= 0, self.alpha * residual, (1 - self.alpha) * residual.abs())
            return self.huber(loss, torch.zeros_like(loss))
        elif tube_type == 'rectangular':
            # Compute L2 norm of the residuals for each component
            y_true = y_true.squeeze(1) # because im a lazy idiot
            residual_x = y_true[:, 0] - y_pred[:, 0]
            residual_y = y_true[:, 1] - y_pred[:, 1]
            norm_residual = torch.sqrt(residual_x**2 + residual_y**2)
            loss = torch.where(norm_residual <= 0, self.alpha * norm_residual, (1 - self.alpha) * norm_residual.abs())
            return self.huber(loss, torch.zeros_like(loss))

def train_and_test(model, criterion, optimizer, train_loader, test_loader, tube_type, alpha, num_epochs=500):
    best_test_loss = float('inf')  # Initialize with a large value

    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0, leave=True):
        model.train()
        train_loss = 0
        for data, targets in tqdm(train_loader, desc="Training Batches", leave=False):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets, tube_type)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Log train loss to wandb
        wandb.log({'Train Loss': train_loss / len(train_loader), 'Epoch': epoch})

        # Evaluate the model
        test_loss, metrics = evaluate_model(model, test_loader, criterion, tube_type)
        
        # Log metrics to wandb
        wandb.log({'Test Loss': test_loss, **metrics, 'Epoch': epoch})
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_path = f"models/{tube_type}/{alpha}/model_epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            wandb.save(model_path)  # Saves the checkpoint to wandb

def evaluate_model(model, test_loader, criterion, tube_type):
    model.eval()
    test_loss = 0
    differences = []
    total_predictions = 0  # Total number of predictions
    count_y_pred_gt_wt1 = 0  # Count of predictions greater than target

    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            test_loss += criterion(outputs, targets, tube_type).item()

            # Increase total predictions
            total_predictions += outputs.size(0)

            if tube_type == 'sphere':
                # Sphere-specific metric calculations
                greater_mask = outputs > targets
                count_y_pred_gt_wt1 += greater_mask.sum().item()
                # Calculate differences where predictions are less than targets
                less_mask = outputs < targets
                differences.extend((targets[less_mask] - outputs[less_mask]).abs().tolist())

            elif tube_type == 'rectangular':
                # Rectangular-specific metric calculations
                targets = targets.squeeze(1) # again... singular braincell human being...
                outputs_x_norm = outputs[:, 0]
                outputs_y_norm = outputs[:, 1]
                targets_x_norm = targets[:, 0]
                targets_y_norm = targets[:, 1]
                
                # Check if either x or y prediction is greater than the target
                greater_mask_x = outputs_x_norm > targets_x_norm
                greater_mask_y = outputs_y_norm > targets_y_norm
                count_y_pred_gt_wt1 += (greater_mask_x | greater_mask_y).sum().item()

                # Calculate differences for x and y separately where predictions are less than targets
                differences_x = (targets_x_norm - outputs_x_norm)[outputs_x_norm < targets_x_norm].abs().tolist()
                differences_y = (targets_y_norm - outputs_y_norm)[outputs_y_norm < targets_y_norm].abs().tolist()
                differences.extend(differences_x)
                differences.extend(differences_y)

    avg_diff = sum(differences) / len(differences) if differences else 0
    test_loss /= len(test_loader)
    proportion_y_pred_gt_wt1 = count_y_pred_gt_wt1 / total_predictions if total_predictions > 0 else 0
    
    metrics = {
        'Test Loss': test_loss,
        'Proportion y_pred > w_{t+1}': proportion_y_pred_gt_wt1,
        'Avg Abs Diff y_pred < w_{t+1}': avg_diff
    }
    return test_loss, metrics

def main(tube_type, alpha, filename):
    # Initialize wandb
    wandb.init()
    config = wandb.config

    X, y = load_and_prepare_data(filename, tube_type)
    train_loader, test_loader = create_data_loaders(X, y, batch_size=64)

    # Set input size based on tube type
    input_size = 6
    output_dim = 2 if tube_type == 'rectangular' else 1

    model = TubeWidthPredictor(input_size=input_size, num_units=config.num_units, num_layers=config.num_layers, output_dim=output_dim)
    criterion = AsymmetricLoss(alpha=config.alpha, delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_and_test(model, criterion, optimizer, train_loader, test_loader, tube_type, alpha, num_epochs=config.num_epochs)

    wandb.finish()

def parse_list(arg):
    return [item.strip() for item in arg.split(',')]

def load_model(model_path, input_size, num_units, num_layers, output_dim):
    model = TubeWidthPredictor(input_size=input_size, num_units=num_units, num_layers=num_layers, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def make_predictions(model, X):
    with torch.no_grad():
        predictions = model(X)
    return predictions

def infer(tube_type, model_path, filename):
    # # Example usage:
    # model_path = 'path/to/model.pth'
    # filename = 'path/to/data.csv'
    # predictions = infer('rectangular', model_path, filename)
    # print(predictions)

    input_size = 6
    num_units = 32
    num_layers = 2
    output_dim = 2 if tube_type == 'rectangular' else 1

    # Load model
    model = load_model(model_path, input_size, num_units, num_layers, output_dim)

    # Load data
    X, _ = load_and_prepare_data(filename, tube_type)  # Ignoring y as it's not needed for inference

    # Make predictions
    predictions = make_predictions(model, X)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the wandb sweep with multiple configurations.")
    parser.add_argument("--alpha", type=parse_list, default="0.8", help="List of alpha values for the loss function.")
    parser.add_argument("--tube_type", type=parse_list, default="sphere", help="List of tube types to process.")
    parser.add_argument("--filename", type=str, default="processed_trajectory_data.csv", help="Filename of the data to process.")

    args = parser.parse_args()

    # Login to wandb
    wandb.login(key="70954bb73c536b7f5b23ef315c7c19b511e8a406")

    # Loop over each combination of alpha and tube_type
    for alpha in args.alpha:
        for tube_type in args.tube_type:
            sweep_config = {
                'method': 'grid',
                'metric': {'name': 'Test Loss', 'goal': 'minimize'},
                'parameters': {
                    'alpha': {'value': float(alpha)},
                    'learning_rate': {'values': [0.001]},
                    'num_units': {'values': [32]},
                    'num_layers': {'values': [2]},
                    'tube_type': {'value': tube_type},
                    'num_epochs': {'value': 500}
                }
            }
            project_name = f"tube_width_experiment_alpha_{alpha}_{tube_type}"
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            wandb.agent(sweep_id, lambda: main(tube_type, float(alpha), args.filename))
