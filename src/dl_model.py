import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os

# ==========================================
# 1. Neural Network Architecture
# ==========================================
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        # We only want the output from the final time step
        final_out = out[:, -1, :] 
        return self.fc(final_out)

# ==========================================
# 2. Sequence Generator
# ==========================================
def create_sequences(data, target_idx, seq_length=5):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 3. Execution Pipeline
# ==========================================
def train_deep_learning():
    print("1. Loading processed data...")
    data_path = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\processed\analysis_ready_data.csv"
    df = pd.read_csv(data_path)
    df = df.sort_values(by=['datetime'])

    features = [
        'hour_sin', 'hour_cos', 'is_weekend', 'is_peak_hour',
        'is_raining', 'low_visibility', 
        'vol_lag_1', 'vol_lag_2', 'vol_lag_48',
        'speed_lag_1', 'speed_lag_2', 'vol_roll_mean_4'
    ]
    target_col = 'traffic_volume'
    
    # We need the target in the matrix to slice it easily
    cols = features + [target_col]
    data_matrix = df[cols].values
    target_idx = len(features) 

    print("2. Scaling Data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)

    print("3. Creating 3D Sequences (Time Steps = 5)...")
    X_seq, y_seq = create_sequences(data_scaled, target_idx, seq_length=5)

    print("4. Time-based Split (80/20)...")
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=False)

    print("5. Initializing LSTM...")
    model = TrafficLSTM(input_size=len(features) + 1, hidden_size=64, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("6. Training Network (5 Epochs)...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"   -> Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    print("7. Evaluating LSTM...")
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()
    
    # Inverse transform the predictions back to real vehicle counts
    # We create a dummy matrix to use the scaler's inverse_transform
    dummy_test = np.zeros((len(preds_scaled), len(cols)))
    dummy_test[:, target_idx] = preds_scaled[:, 0]
    preds_real = scaler.inverse_transform(dummy_test)[:, target_idx]
    
    dummy_actual = np.zeros((len(y_test), len(cols)))
    dummy_actual[:, target_idx] = y_test
    actual_real = scaler.inverse_transform(dummy_actual)[:, target_idx]

    rmse = np.sqrt(mean_squared_error(actual_real, preds_real))
    mape = mean_absolute_percentage_error(actual_real, preds_real) * 100

    print("\n==============================")
    print("      DEEP LEARNING SCORE     ")
    print("==============================")
    print(f"LSTM Model | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
    print("==============================\n")
    
    # Save the model
    model_dir = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\models"
    torch.save(model.state_dict(), os.path.join(model_dir, "lstm_model.pth"))
    print(f"Deep learning model saved to {model_dir}")

if __name__ == "__main__":
    train_deep_learning()