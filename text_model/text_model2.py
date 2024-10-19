import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pandas as pd
from pathlib import Path
import numpy as np
from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "BrainStrokeFinalModel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)



# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = StrokeMLP().to(device)

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_0.eval()

# New sample for prediction (example values)
new_sample = torch.tensor([0, 67.0, 0, 1, 0, 0, 0, 228.69, 36.6, 0], dtype=torch.float32).to(device)

# Scale the new sample using the same scaler
new_sample_np = scaler.transform(new_sample.cpu().numpy().reshape(1, -1))
new_sample_tensor = torch.tensor(new_sample_np, dtype=torch.float32).to(device)
new_sample_tensor = new_sample_tensor.view(1, -1)  # Reshape to [1, 10]

# Make the prediction with the loaded model
with torch.no_grad():
    output = loaded_model_0(new_sample_tensor)
    predicted_probability = torch.sigmoid(output).item()
    predicted_class = 1 if predicted_probability >= 0.5 else 0

print(f"Predicted Probability: {predicted_probability:.4f}")
print(f"Predicted Class: {predicted_class}")