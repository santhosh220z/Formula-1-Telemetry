import argparse
from train import train_model

TRACKS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", 
    "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada", 
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands", 
    "Italy", "Singapore", "United States", "Mexico", "Brazil", 
    "Las Vegas", "Qatar", "Abu Dhabi"
]

class Args:
    pass

def main():
    year = 2023 # Defaults to 2023 since it has complete historical data
    for track in TRACKS:
        print(f"\n{'='*50}\nStarting Training for: {track} ({year})\n{'='*50}")
        args = Args()
        args.year = year
        args.event = track
        args.epochs = 50
        args.batch_size = 32
        args.lr = 0.001
        args.hidden_size = 64
        args.num_layers = 2
        args.dropout = 0.2
        
        try:
            train_model(args)
        except Exception as e:
            print(f"Failed to train for {track}: {e}")

if __name__ == "__main__":
    main()
