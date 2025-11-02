import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1️⃣ Load dataset ---
df = pd.read_csv("cubeSat_battery_normal.csv")
features = ["temperature_C", "voltage_V", "current_A"]
X = df[features].values

# --- 2️⃣ Normalize data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# --- 3️⃣ Create train & validation datasets ---
dataset = TensorDataset(X_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 4️⃣ Define Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 3)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 5️⃣ Initialize model, loss, optimizer ---
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 6️⃣ Train the model ---
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            output = model(x)
            val_loss += criterion(output, x).item()

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss/len(train_loader):.6f} - Val Loss: {val_loss/len(val_loader):.6f}")

# --- 7️⃣ Compute threshold using all training data ---
model.eval()
train_errors = []
with torch.no_grad():
    for x in X_tensor:
        x = x.unsqueeze(0)
        reconstructed = model(x)
        loss = nn.MSELoss()(reconstructed, x).item()
        train_errors.append(loss)

threshold = np.mean(train_errors) + 3 * np.std(train_errors)
print("✅ Recommended anomaly threshold:", threshold)

# --- 8️⃣ Save model and scaler ---
torch.save(model.state_dict(), "autoencoder_battery.pth")
joblib.dump(scaler, "battery_scaler.pkl")
print("✅ Training complete — model and scaler saved!")
