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
    n = feature_matrix.shape[0]
    r = range(n)
    for t in range(T):
        for i in r:
            updates = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta0)
            theta = updates[0]
            theta0 = updates[1]
            theta_updates = theta_updates + theta
            theta0_updates = theta0_updates + theta0
    return (theta_updates / (n*T), theta0_updates / (n*T))


feat = np.array([[-0.15003148, -0.14416684, -0.1262574 ,  0.08673096, -0.10171562,  0.44435962, 0.09793402,  0.06426548,  0.2068302 , -0.25101512],[ 0.48525363, -0.13973322,  0.32399485,  0.09038688, -0.01777904,  0.20302376,0.00573383,  0.27411173, -0.49365211, -0.32070898],[-0.09755884, -0.38786729,  0.31660232,  0.47883757, -0.40688115, -0.16919978,0.1017912 , -0.09556   ,  0.35575884, -0.18884246],[-0.0961716 , -0.3337901 ,  0.14613577,  0.44526045, -0.34122835,  0.36526918,-0.3763469 , -0.31772996, -0.17860048,  0.0129932 ],[-0.43964534, -0.34744255, -0.42975444, -0.00341418,  0.01457779,  0.13564628,-0.00963358, -0.40743617,  0.00100382,  0.21690422]])
lab = np.array([-1,  1,  1,  1,  1])

new_iteration = average_perceptron(feat, lab, 5)
print(new_iteration)

# expected theta ['-0.0899358', '-0.5325342', '0.0804505', '0.2032610', '-0.0627960', '-0.7780425', '-0.1485239', '-0.6245247', '-0.4656467', '0.5302950']