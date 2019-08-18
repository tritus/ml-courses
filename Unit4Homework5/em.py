import numpy as np
from scipy.stats import norm as norm_dist

X = np.array([-1,0,4,5,6])
p1 = 0.5
p2 = 0.5
m1 = 6
m2 = 7
sd1 = 1
sd2 = 2

x_from_1 = norm_dist.pdf(X, loc=m1, scale=sd1)
x_from_2 = norm_dist.pdf(X, loc=m2, scale=sd2)
x_likelihood = np.array([x_from_1, x_from_2])
weights = np.array([p1, p2])

likelihoods = np.matmul(x_likelihood.transpose(), weights)
log_likelihoods = np.log(likelihoods)
log_likelihood = np.sum(log_likelihoods)

print("log likelihood : " + str(log_likelihood))

gaussian_likelihood_1 = p1*norm_dist.pdf(X, loc=m1, scale=sd1)/(p1*norm_dist.pdf(X, loc=m1, scale=sd1)+p2*norm_dist.pdf(X, loc=m2, scale=sd2))
print("likelihood for gaussian 1: \n" + str(gaussian_likelihood_1))

gaussian_likelihood_2 = p2*norm_dist.pdf(X, loc=m2, scale=sd2)/(p1*norm_dist.pdf(X, loc=m1, scale=sd1)+p2*norm_dist.pdf(X, loc=m2, scale=sd2))
print("likelihood for gaussian 1: \n" + str(gaussian_likelihood_2))

x_1 = np.array([5,6])
x_2 = np.array([-1,0,4])

gammas_1 = np.array([0.6667,0.6938])
gammas_2 = np.array([0.99999986, 0.99998608, 0.54533833])

m1 = np.matmul(x_1, gammas_1.transpose())/np.sum(gammas_1)
m2 = np.matmul(x_2, gammas_2.transpose())/np.sum(gammas_2)
print("new m1 : " + str(m1) + " and m2 : " + str(m2))

v1 = np.matmul((x_1-m1)**2, gammas_1.transpose())/np.sum(gammas_1)
v2 = np.matmul((x_2-m2)**2, gammas_2.transpose())/np.sum(gammas_2)
new_sd1 = v1**(0.5)
new_sd2 = v2**(0.5)
print("new sd1 grew by : " + str(new_sd1/sd1) + " and sd2 by : " + str(new_sd2/sd2))