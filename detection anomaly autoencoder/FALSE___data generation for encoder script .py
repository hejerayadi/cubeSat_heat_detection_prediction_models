import numpy as np
import pandas as pd

# --- Simulation parameters ---
duration_s = 3600
sampling_interval = 5
time = np.arange(0, duration_s, sampling_interval)

# --- Simulate anomalies ---
# Battery overheating + voltage drop + current spikes
temp = 25 + 10 * np.sin(time / 600) + np.random.normal(0, 0.4, len(time))
temp[400:600] += 15   # overheating event
temp[900:1000] += 8   # smaller temp anomaly

voltage = 12.6 - 0.002 * (time / 60) + np.random.normal(0, 0.03, len(time))
voltage[400:600] -= 0.8   # voltage sag during overheating
voltage[900:1000] -= 0.5
voltage = np.clip(voltage, 9.5, 12.6)

current = 0.5 + 0.3 * np.sin(time / 120) + np.random.normal(0, 0.1, len(time))
current[400:600] += 0.7  # high current during fault
current = np.clip(current, 0.1, 2.0)

# --- Convert to timestamps ---
start_time = pd.Timestamp("2025-11-02 15:00:00")
timestamps = [start_time + pd.Timedelta(seconds=int(t)) for t in time]

# --- Combine into DataFrame ---
fault_df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_C": temp.round(2),
    "voltage_V": voltage.round(3),
    "current_A": current.round(3)
})

fault_df.to_csv("cubeSat_battery_faulty.csv", index=False)
print("⚠️ Faulty dataset saved as cubeSat_battery_faulty.csv with", len(fault_df), "samples")
