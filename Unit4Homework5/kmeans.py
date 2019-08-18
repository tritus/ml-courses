import numpy as np

def clusters(initial_z, x):
	print("Initializing with \nz : \n" + str(initial_z) + "\nx : \n" + str(x))
	z = initial_z
	K = z.shape[0]
	labels = np.zeros(x.shape[0])
	previous_z = np.zeros(z.shape)
	while are_not_same_set(previous_z,z):
		previous_z = z
		labels = x_labels(x, z)
		z = new_centers_of_clusters(labels, x, K)
		print("Clusters have been updated to\n" + str(z))
	return z, x_labels(x, z)

def are_not_same_set(set1, set2):
	for i in range(set1.shape[0]):
		if set1[i] not in set2:
			return False
	return True

def l1_single_cost(xi, zj):
	diff = xi-zj
	return np.sum(diff)**2

def x_labels(x, z):
	labels = []
	for i in range(x.shape[0]):
		xi = x[i]
		labels += [xi_label(xi, z)]
	return np.array(labels)

def xi_label(xi, z):
	final_j = 0
	for j in range(z.shape[0]):
		current_z = z[j]
		if l1_single_cost(xi, current_z) > l1_single_cost(xi, z[final_j]):
			final_j = j
	return final_j

def new_centers_of_clusters(labels, x, K):
	z = []
	for j in range(K):
		z += [new_center_of_cluster(labels, x, j)]
	return np.array(z)

def new_center_of_cluster(labels, x, j):
	selected_xs = select_labeled_xs(x, labels, j)
	x_sum = sum_vectors(selected_xs)
	return x_sum / selected_xs.shape[0]

def sum_vectors(vectors):
	vsum = np.zeros(vectors.shape[1])
	for i in range(vectors.shape[0]):
		vsum += vectors[i]
	return vsum

def select_labeled_xs(x, labels, j):
	selection = []
	for xi, label in zip(x,labels):
		if label == j:
			selection += [xi]
	return np.array(selection)

def label_selector(labels, j, d):
	selector = []
	for i in range(labels.shape[0]):
		label = labels[i]
		selector += [toto(label, j, d)]
	return selector

def toto(label, j, d):
	if label == j:
		return np.identity(d)
	else:
		return np.zeros((d,d))

X = np.array([[0,-6],[4,4],[0,0],[-5,2]])
Z = np.array([[-5,2],[0,-6]])

computed_clusters = clusters(Z, X)

print("Output : \n" + str(computed_clusters))