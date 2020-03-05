import math
import matplotlib.pyplot as plt

def tent(x):
  if 0 <= x <= 0.5:
    return 2 * x
  return (2 - 2 * x)

# x = math.pi / 7
# X = []
# Y = []
# for i in range(40):
#   X.append(i)
#   Y.append(x)
#   x = tent(x)
#
# plt.plot(X, Y)
# plt.show()

def logistic(a, x):
  return a * x * (1-x)

def lyapunov(fn, *args, **kwargs):
  sum
  for i in range(100):
