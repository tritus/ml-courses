import numpy as np

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    epsilon = 0.001
    theta = current_theta
    theta0 = current_theta_0
    computed_label = np.matmul(theta, feature_vector) + theta0
    score = label * computed_label
    if score < epsilon:
        theta = theta + label * feature_vector
        theta0 = theta0 + label
    return (theta, theta0)

def average_perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1])
    theta0 = 0
    theta_updates = np.zeros(feature_matrix.shape[1])
    theta0_updates = 0
    r = range(1)
    n = 1
    for t in range(T):
        for i in r:
            updates = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta0)
            theta_updates = theta_updates + updates[0]
            theta0_updates = theta0_updates + updates[1]
    return (theta_updates / (n*T), theta0_updates / (n*T))


feat = np.array([[1,0]])
lab = np.array([1])
cur_th = np.array([0,-1])
cur_th0 = 0

new_iteration = average_perceptron(feat, lab, 10)
print(new_iteration)