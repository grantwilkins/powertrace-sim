import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

power_trace = pd.read_csv(
    "/Users/grantwilkins/powertrace-sim/client/llama-3-8b/llama-3-8b_tp1_p0.5_d2025-03-13-18-28-09.csv",
    skipinitialspace=True,
)
power_trace["memory.used [MiB]"] = (
    power_trace["memory.used [MiB]"].replace("MiB", "", regex=True).astype(float)
)
power_trace["power.draw [W]"] = (
    power_trace["power.draw [W]"].replace("W", "", regex=True).astype(float)
)
power_trace = power_trace[power_trace["memory.used [MiB]"] > 20]
# power_trace["power_sum"] = power_trace["power.draw [W]"].rolling(8).sum().fillna(0)


# calculate z score of power draw
# print(power_trace["power_sum"])
# power_trace = power_trace[np.abs(stats.zscore(power_trace["power_sum"])) < 3]
# power_trace = power_trace[power_trace["memory.used [MiB]"] > 20]
# power_trace["time [s]"] = np.arange(0, len(power_trace) * 0.25, 0.25)
# plt.plot(power_trace["time [s]"], power_trace["power_sum"])
# plt.show()

power_trace = power_trace[np.abs(stats.zscore(power_trace["power.draw [W]"])) < 3]
power_trace["time [s]"] = np.arange(0, len(power_trace) * 0.25, 0.25)
plt.plot(power_trace["time [s]"], power_trace["power.draw [W]"])
plt.show()
