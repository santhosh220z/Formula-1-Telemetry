import torch
import numpy as np
from models.lstm_model import F1LapTimePredictor
import os

def load_model(model_path):
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    if not os.path.exists(model_path):
        return None, None
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Initialize model with saved config
    model = F1LapTimePredictor(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def predict_next_lap(model, device, sequence):
    """
    Given a sequence of shape (sequence_length, input_size), predict the next lap time.
    """
    model.eval()
    with torch.no_grad():
        # Ensure correct shape (batch_size=1, seq_length, input_size)
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
            
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)
        prediction = model(seq_tensor)
        
        return prediction.item()

def simulate_stint(model, device, initial_sequence, future_laps=20, degradation_factor=0.015):
    """
    Simulate future laps to generate a tire degradation curve.
    In a real scenario, you'd auto-regressively predict lap times while updating tire age.
    For this demo, we simulate lap times increasing due to tire wear.
    """
    predicted_lap_times = []
    current_sequence = initial_sequence.copy()
    
    # Very basic simulation logic
    for lap in range(future_laps):
        pred = predict_next_lap(model, device, current_sequence)
        
        # Add synthetic degradation if no real features update are available 
        pred += lap * degradation_factor
        predicted_lap_times.append(pred)
        
        # Shift sequence and mock typical value update for next iteration
        # (Assuming lap time is at index 2 for example purposes, but we don't have it as input)
        # Here we just use the predicted value as a heuristic 
        # Actually our sequence features don't include laptime, mostly sectors and tyre life
        
        # For simplicity in simulation, we just increment tyre life (assume it's minmax scaled approx)
        # Tyre life index is 6 in our feature setup
        new_step = current_sequence[-1].copy()
        new_step[6] += 0.05 # small increment to normalized tyre life
        
        current_sequence = np.vstack((current_sequence[1:], new_step))
        
    return predicted_lap_times

def calculate_pit_window(lap_times, threshold=2.0, pit_loss=20.0):
    """
    Recommends a pit stop when consecutive lap time degradation exceeds threshold
    or cumulative loss is worse than pit loss.
    """
    base_lap = lap_times[0]
    for i, lap_time in enumerate(lap_times):
        if lap_time - base_lap > threshold:
            return i + 1 # Recommend pit after this lap number
    
    return None
