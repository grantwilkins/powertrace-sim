import pandas as pd
import matplotlib.pyplot as plt

# Make plot for burn plot timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB] power for every 8 values
df = pd.read_csv("burn.csv")

# convert timestamp to seconds 2025/05/05 20:58:44.813
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].astype(int) / 10**9
# convert power.draw [W] to float
df[" power.draw [W]"] = df[" power.draw [W]"].str.replace(" W", "")
df[" power.draw [W]"] = df[" power.draw [W]"].astype(float)

# plot every value every 8 values

df = df.iloc[::8, :]
# plot power.draw [W], utilization.gpu [%], memory.used [MiB]
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df[" power.draw [W]"], label="Power draw [W]")
plt.show()
