import torch
import torch.nn as nn
import numpy as np
import joblib

# ---- 1️⃣ Autoencoder definition (exactly as trained) ----
class Autoencoder(nn.Module):
    def __init__(self, input_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
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
            nn.Linear(4, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---- 2️⃣ Load trained model ----
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder_battery.pth", map_location="cpu"))
model.eval()

# ---- 3️⃣ Load scaler ----
scaler = joblib.load("battery_scaler.pkl")

# ---- 4️⃣ Define anomaly threshold from training ----
threshold = 1.3808  # from your last training run

# ---- 5️⃣ Define test samples ----
normal_sample = np.array([[20.0, 12.0, 0.6]], dtype=np.float32)
abnormal_sample = np.array([[75.0, 9.0, 1.8]], dtype=np.float32)

# ---- 6️⃣ Function to test one sample ----
def test_sample(sample, label):
    # Scale the sample
    sample_scaled = scaler.transform(sample)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
    # Get reconstruction
    with torch.no_grad():
        reconstructed = model(sample_tensor)
        loss = nn.MSELoss()(reconstructed, sample_tensor).item()
    print(f"\n{label}: Reconstruction Error = {loss:.6f}")
    if loss > threshold:
        print("🚨 Anomaly detected!")
    else:
        print("✅ Normal sample")

# ---- 7️⃣ Run tests ----
test_sample(normal_sample, "Normal Sample")
test_sample(abnormal_sample, "Abnormal Sample")
