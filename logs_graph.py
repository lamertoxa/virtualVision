import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

# Path to your JSON file
json_file_path = 'logs/log_gpu.json'

# Read data from JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Convert timestamp strings to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the timestamp column as the index
df.set_index('timestamp', inplace=True)

# Convert GPU temperature to numeric values (remove ℃)
df['gpu_temp'] = pd.to_numeric(df['gpu_temp'], errors='coerce')

# Plotting
plt.figure(figsize=(10, 5))

# Plot GPU memory utilization
plt.subplot(2, 2, 1)
plt.plot(df.index, df['gpu_memory_utilization'], label='GPU Memory Utilization')
plt.ylabel('Utilization (%)')
plt.title('GPU Memory Utilization')
plt.legend()

# Plot GPU temperature
plt.subplot(2, 2, 2)
plt.plot(df.index, df['gpu_temp'], label='GPU Temperature', color='red')
plt.ylabel('Temperature (℃)')
plt.title('GPU Temperature')
plt.legend()

# Plot System Memory Utilization
plt.subplot(2, 2, 3)
plt.plot(df.index, df['system_memory_utilization'], label='System Memory Utilization', color='green')
plt.ylabel('Utilization (%)')
plt.title('System Memory Utilization')
plt.legend()

# Plot CPU Utilization
plt.subplot(2, 2, 4)
plt.plot(df.index, df['cpu_utilization'], label='CPU Utilization', color='purple')
plt.ylabel('Utilization (%)')
plt.title('CPU Utilization')
plt.legend()

plt.tight_layout()
plt.show()
