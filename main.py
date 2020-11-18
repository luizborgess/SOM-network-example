import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 8], [2, 7], [3, 7], [6, 3], [7, 3], [8, 4]], np.int32)
np.random.shuffle(x)

np.random.shuffle(x)
print(x.shape)

# (b - a) * random_sample() + a
w = 7 * np.random.random_sample((2, 2)) + 1

print(w)
plt.ion()


def argmin(w, xi):
    d = np.zeros(w.shape[0])
    for i, wi in enumerate(w):
        d[i] = np.linalg.norm(wi - xi)

    return np.argmin(d)



alpha = 0.1

for i in range(20):

    for id, xi in enumerate(x):
        n = argmin(w, xi)

        w[n] = w[n] + (xi - w[n]) * alpha

        plt.scatter(x[:, 0], x[:, 1], color="blue")
        plt.scatter(w[:, 0], w[:, 1], color="red")
        plt.pause(0.001)
        plt.clf()

plt.show()
