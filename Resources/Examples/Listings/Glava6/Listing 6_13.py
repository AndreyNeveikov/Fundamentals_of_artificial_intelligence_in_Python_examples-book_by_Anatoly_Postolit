# Listing 6.13
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker=".")
plt.show()
