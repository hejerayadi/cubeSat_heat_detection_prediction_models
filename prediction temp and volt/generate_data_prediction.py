import numpy as np
import pandas as pd

# -----------------------
# Parameters
# -----------------------
n_samples = 2000       # number of time steps
dt = 1                 # seconds per step
temp_base = 25         # initial temperature (°C)
volt_base = 4.2        # initial voltage (V)
curr_base = 0.5        # initial current (A)

# -----------------------
# Initialize arrays
# -----------------------
temperature = [temp_base]
voltage = [volt_base]
current = [curr_base]

# -----------------------
# Generate synthetic time-series
# -----------------------
for i in range(1, n_samples):
    # Temperature trend + noise
    temp_change = np.random.normal(0, 0.05) + 0.001*(i)   # slow upward trend
    temperature.append(temperature[-1] + temp_change)
    
    # Voltage decay (discharge) + noise
    volt_change = -0.0005 + np.random.normal(0, 0.001)
    voltage.append(voltage[-1] + volt_change)
    
    # Current fluctuation
    curr_change = np.random.normal(0, 0.01)
    current.append(max(0, current[-1] + curr_change))  # current can't be negative

# -----------------------
# Create DataFrame
# -----------------------
df = pd.DataFrame({
    'timestamp': np.arange(n_samples)*dt,
    'temperature': temperature,
    'voltage': voltage,
    'current': current
})

# -----------------------
# Derived features
# -----------------------
df['power'] = df['voltage'] * df['current']
df['delta_temp'] = df['temperature'].diff().fillna(0)
df['delta_voltage'] = df['voltage'].diff().fillna(0)
df['delta_current'] = df['current'].diff().fillna(0)

# -----------------------
# Next-step targets (y)
# -----------------------
df['temperature_next'] = df['temperature'].shift(-1)
df['voltage_next'] = df['voltage'].shift(-1)
df['current_next'] = df['current'].shift(-1)

# Remove last row since it has no next-step target
df = df[:-1]

# -----------------------
# Save to CSV
# -----------------------
df.to_csv('synthetic_battery_prediction_data.csv', index=False)
print("Data generated successfully! Here's a preview:")
print(df.head())
