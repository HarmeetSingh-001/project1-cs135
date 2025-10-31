import numpy as np
import matplotlib.pyplot as plt

# Probability values
p = np.linspace(0.001, 0.999, 500)

# Log loss for y=1 and y=0
loss_y1 = -np.log(p)
loss_y0 = -np.log(1 - p)

# Plot
plt.figure(figsize=(8,5))
plt.plot(p, loss_y1, label="y = 1", color='blue')
plt.plot(p, loss_y0, label="y = 0", color='red')
plt.title("Log Loss vs Predicted Probability")
plt.xlabel("Predicted Probability (p̂)")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
