import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.data_processing import prepare_dataset
from models.lstm_model import F1LapTimePredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading data for {args.event} {args.year}...")
    try:
        X, y, scaler, le, df = prepare_dataset(args.year, args.event, sequence_length=5)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    if len(X) == 0:
        print("Not enough data to train.")
        return

    # Replace NaNs in target if any
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Convert features to numeric and handle NaNs safely
    X = np.nan_to_num(X.astype(np.float32), nan=0.0)
    y = np.nan_to_num(y.astype(np.float32), nan=0.0)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize Model
    input_size = X.shape[2]
    model = F1LapTimePredictor(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    # 3. Define Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(test_loader.dataset)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 5. Final Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    print("\n--- Final Evaluation ---")
    print(f"MAE:  {mae:.4f} seconds")
    print(f"RMSE: {rmse:.4f} seconds")
    print(f"R²:   {r2:.4f}")

    # 6. Save Model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f'lstm_{args.year}_{args.event}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }, model_path)
    
    # Save config text for app reference
    with open("models/latest_model_info.txt", "w") as f:
        f.write(f"model_path={model_path}\n")
        f.write(f"input_size={input_size}\n")
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--event", type=str, default="Bahrain")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    args = parser.parse_args()
    train_model(args)
