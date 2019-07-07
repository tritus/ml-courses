import numpy as np

class MatricPredictorWithFactorization:
	def __init__(self, known_values, max_iterations):
		self.known_values = known_values
		self.max_iterations = max_iterations

	def predicted_matrix(self):
		for t in range(2*self.max_iterations):
			for i in 

