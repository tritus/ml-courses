import numpy as np

xs = np.array([
  [1,0,1],
  [1,1,1],
  [1,1,-1],
  [-1,1,1]
])

ys = np.array([
  2,
  2.7,
  -0.7,
  2
])

theta = np.array([0,1,2])

print("##########")
print("Hinge Loss")
print("##########")

loss = 0
items_count = xs.shape[0]
for i in range(items_count):
  difference_in_expectation = ys[i] - np.matmul(xs[i], theta)
  if difference_in_expectation < 1:
    loss += 1 - difference_in_expectation

total_loss = loss / items_count

print("computed loss : " + str(total_loss))

print("##########")
print("Squared Error")
print("##########")

loss = 0
items_count = xs.shape[0]
for i in range(items_count):
  difference_in_expectation = ys[i] - np.matmul(xs[i], theta)
  loss += difference_in_expectation ** 2 / 2

total_loss = loss / items_count

print("computed loss : " + str(total_loss))

