import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

def test_seeds(K):
	print("\n############## K=" + str(K) + " ###############")

	mixture0, post0 = common.init(X,K,0)
	mixture1, post1 = common.init(X,K,1)
	mixture2, post2 = common.init(X,K,2)
	mixture3, post3 = common.init(X,K,3)
	mixture4, post4 = common.init(X,K,4)

	cost0 = kmeans.run(X,mixture0,post0)[2]
	cost1 = kmeans.run(X,mixture1,post1)[2]
	cost2 = kmeans.run(X,mixture2,post2)[2]
	cost3 = kmeans.run(X,mixture3,post3)[2]
	cost4 = kmeans.run(X,mixture4,post4)[2]

	print("K=" + str(K) + " seed=0 : cost=" + str(cost0))
	print("K=" + str(K) + " seed=1 : cost=" + str(cost1))
	print("K=" + str(K) + " seed=2 : cost=" + str(cost2))
	print("K=" + str(K) + " seed=3 : cost=" + str(cost3))
	print("K=" + str(K) + " seed=4 : cost=" + str(cost4))

test_seeds(1)
test_seeds(2)
test_seeds(3)
test_seeds(4)

