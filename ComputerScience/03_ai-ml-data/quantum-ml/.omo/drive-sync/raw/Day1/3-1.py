import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 500)

y = x**4 - 3*x**2 + x

plt.figure(figsize=(8,5))

plt.plot(x, y)

plt.title("Loss Surface Example")
plt.xlabel("Parameter")
plt.ylabel("Loss")

plt.grid()

plt.show()