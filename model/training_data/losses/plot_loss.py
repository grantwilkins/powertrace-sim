import matplotlib.pyplot as plt
import numpy as np

x1 = np.load(
    "/Users/grantwilkins/powertrace-sim/model/training_data/losses/training_losses_deepseek-r1-distill-8b_h100_tp1.npy"
)
x2 = np.load(
    "/Users/grantwilkins/powertrace-sim/model/training_data/losses/training_losses_deepseek-r1-distill-8b_h100_tp2.npy"
)
x4 = np.load(
    "/Users/grantwilkins/powertrace-sim/model/training_data/losses/training_losses_deepseek-r1-distill-8b_h100_tp4.npy"
)
x8 = np.load(
    "/Users/grantwilkins/powertrace-sim/model/training_data/losses/training_losses_deepseek-r1-distill-8b_h100_tp8.npy"
)
plt.figure(figsize=(5, 3))
plt.plot(x1, label="TP=1")
plt.plot(x2, label="TP=2")
plt.plot(x4, label="TP=4")
plt.plot(x8, label="TP=8")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 500)
plt.ylim(0, 1.75)
plt.savefig(
    "/Users/grantwilkins/powertrace-sim/model/training_data/losses/training_losses_deepseek-r1-distill-8b_h100.pdf"
)
