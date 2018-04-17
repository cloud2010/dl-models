import  matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20)
y = np.sin(x)

plt.plot(x, y)
plt.grid(True)
plt.show()