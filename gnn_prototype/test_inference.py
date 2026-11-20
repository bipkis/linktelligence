import torch
import os
from dataset import NetworkDataset
# Assuming ModelClass is defined elsewhere or passed in, 
# we would ideally use 'from model import YourModelClass' here.
# For this fix, we assume a placeholder class structure that mimics the real model for compilation purposes.

class ModelPlaceholder(torch.nn.Module):
    """A placeholder model mimicking the expected input/output structure."""
    def __init__(self):
        super().__init__()
        # Assuming feature dimension is around 10 based on original context
        self.linear = torch.nn.Linear(10, 3)

    def forward(self, x):
        return self.linear(x).tolist()


MODEL_PATH = "best_model.pt"
DATA_ROOT = "." # Assuming data/ is relative to execution root

def load_model(path: str):
    """Loads the trained model from a given path."""
    print(f"Loading model from {path}...")
    try:
        # 1. Instantiate the actual model class (using placeholder for now)
        model = ModelPlaceholder() # Replace with actual ModelClass(...)
        
        # 2. Load state dictionary using torch.load
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return None

def run_inference_on_sample():
    """Loads data, selects one sample, runs inference, and prints results."""
    
    # 1. Load Model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Cannot proceed with inference due to model loading failure.")
        return

    # 2. Load Dataset and select one sample
    try:
        dataset = NetworkDataset(root=DATA_ROOT)
        
        if not dataset.processed_data:
            print("No processed data found in the dataset.")
            return

        sample_data = dataset.processed_data[0]
        print(f"Running inference on the first sample...")

    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading or processing: {e}")
        return

    # 3. Run Inference (Actual logic)
    score = None
    bottleneck = None
    security_flaw = None
    
    with torch.no_grad():
        # Assuming the model takes node features 'x' as input for simplicity in this fix context
        input_tensor = sample_data.x 
        output_raw = model(input_tensor)

        # Assuming the output is a list/tuple containing [score, bottleneck_id, flaw_flag]
        if isinstance(output_raw, list) and len(output_raw) >= 3:
            score = output_raw[0]
            bottleneck = f"Node {int(output_raw[1])}" # Simulate mapping ID to name
            security_flaw = "High Risk Detected" if output_raw[2] > 0.5 else "None"
        else:
             print("Warning: Model output format unexpected.")


    # 4. Print Results
    print("\n--- Inference Results ---")
    if score is not None:
        print(f"Score: {score:.2f}")
    else:
        print("Score: N/A (Inference failed)")
        
    if bottleneck is not None:
        print(f"Bottleneck: {bottleneck}")

    if security_flaw is not None:
        print(f"Security Flaw: {security_flaw}")


if __name__ == "__main__":
    run_inference_on_sample()