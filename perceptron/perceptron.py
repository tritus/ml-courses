import numpy as np

class Perceptron():

  def __init__(self, test_set):
    self.test_set = test_set

  def test_match(self, test_data, theta, theta0):
    return test_data[1] * (np.matmul(theta, test_data[0]) + theta0)

  def all_good(self, theta, theta0):
    results = []
    any_wrong_label = False
    for test_data in self.test_set:
      results += [self.test_match(test_data, theta, theta0)]
    for result in results:
      any_wrong_label = any_wrong_label or result <= 0
    return any_wrong_label == False

  def optimal_theta(self):
    theta = np.zeros(self.test_set[0][0].shape)
    theta0 = 0
    while (self.all_good(theta, theta0) == False):
      for i in range(len(self.test_set)):
        test_data = self.test_set[i]
        if self.test_match(test_data, theta, theta0) <= 0:
          theta = theta + test_data[1] * test_data[0]
          theta0 = theta0 + test_data[1]
          print("temporary theta : " + str(theta) + " and theta0 : " + str(theta0))
    return [theta, theta0]




test_set = [
  np.array([np.array([0,0,0]), np.array(+1)]),
  np.array([np.array([0,0,1]), np.array(-1)]),
  np.array([np.array([0,1,0]), np.array(-1)]),
  np.array([np.array([0,1,1]), np.array(-1)]),
  np.array([np.array([1,0,0]), np.array(-1)]),
  np.array([np.array([1,0,1]), np.array(-1)]),
  np.array([np.array([1,1,0]), np.array(-1)]),
  np.array([np.array([1,1,1]), np.array(-1)])
]

perceptron = Perceptron(test_set)
optimal_theta = perceptron.optimal_theta()
print("final theta : " + str(optimal_theta[0]) + " and theta0 : " + str(optimal_theta[1]))