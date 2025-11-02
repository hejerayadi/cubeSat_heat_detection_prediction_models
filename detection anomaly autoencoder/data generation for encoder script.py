import numpy as np
import pandas as pd

# --- Simulation parameters ---
duration_s = 3600              # total duration: 1 hour (in seconds)
sampling_interval = 5          # one sample every 5 seconds
time = np.arange(0, duration_s, sampling_interval)

# --- Simulate sensor values ---
# Temperature increases slowly with small random variation
temp = 25 + 8 * np.sin(time / 800) + np.random.normal(0, 0.2, len(time))

# Voltage drops slightly over time with small random noise
voltage = 12.6 - 0.0015 * (time / 60) + np.random.normal(0, 0.02, len(time))
voltage = np.clip(voltage, 10.8, 12.6)  # keep values in a realistic range

# Current oscillates to simulate load changes, plus some random noise
current = 0.5 + 0.3 * np.sin(time / 120) + np.random.normal(0, 0.05, len(time))
current = np.clip(current, 0.1, 1.5)

# --- Convert time to proper timestamps ---
start_time = pd.Timestamp("2025-11-02 14:00:00")
timestamps = [start_time + pd.Timedelta(seconds=int(t)) for t in time]

# --- Combine into a DataFrame ---
df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_C": temp.round(2),
    "voltage_V": voltage.round(3),
    "current_A": current.round(3)
})

# --- Save to CSV ---
df.to_csv("cubeSat_battery_normal.csv", index=False)
print(f"✅ Dataset saved as cubeSat_battery_normal.csv with {len(df)} samples")
