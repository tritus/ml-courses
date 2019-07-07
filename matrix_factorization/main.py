import numpy as np

x = np.array([[24, 12, 6], [0, 0, 0], [12, 6, 3], [24, 12, 6]])
y = np.array([[5, None, 7], [None, 2, None], [4, None, None], [None, 3, 6]])

u = np.array([6, 0, 3, 6])
v = np.array([4, 2, 1])

l = 1

total_error = 0
total_regularization = 0

for i in range(x.shape[0]):
	for j in range(x.shape[1]):
		x_elem = x[i][j]
		y_elem = y[i][j]
		if (y_elem is not None):
			total_error += 1/2*(y_elem-x_elem)**2
		total_regularization += l/2*x_elem**2

print(total_error)
print(total_regularization)

total_regularization_from_uv = l/2*np.linalg.norm(u)**2 + l/2*np.linalg.norm(v)**2
print(total_regularization_from_uv)

for i in range(u.shape[0]):
	residual = 0
	coeff = 0
	for j in range(y[i].shape[0]):
		y_elem = y[i][j]
		v_elem = v[j]
		if (y_elem is not None):
			residual += y_elem * v_elem
			coeff += v_elem * v_elem
	coeff += l
	solution = residual/coeff
	print("U_" + str(i + 1) + " = " + str(solution))
