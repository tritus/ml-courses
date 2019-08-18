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

	naive_em_estimate0 = naive_em.run(X,mixture0,post0)
	naive_em_estimate1 = naive_em.run(X,mixture1,post1)
	naive_em_estimate2 = naive_em.run(X,mixture2,post2)
	naive_em_estimate3 = naive_em.run(X,mixture3,post3)
	naive_em_estimate4 = naive_em.run(X,mixture4,post4)

	print("K=" + str(K) + " seed=0 : likelihood=" + str(naive_em_estimate0[2]))
	print("K=" + str(K) + " seed=1 : likelihood=" + str(naive_em_estimate1[2]))
	print("K=" + str(K) + " seed=2 : likelihood=" + str(naive_em_estimate2[2]))
	print("K=" + str(K) + " seed=3 : likelihood=" + str(naive_em_estimate3[2]))
	print("K=" + str(K) + " seed=4 : likelihood=" + str(naive_em_estimate4[2]))


# K mean initialization

test_seeds(1)
test_seeds(2)
test_seeds(3)
test_seeds(4)

# EM algo
print("############## EM Algorythme implemented ###############")
mixture, post = common.init(X,3,0)
naive_em_estimate = naive_em.run(X,mixture,post)[2]
print("naive EM log likelihood : " + str(naive_em_estimate))

print("############## Some Tests ######################")
initialMixture, initialPost = common.init(X,1,0)
mixtureK, postK, distorision = kmeans.run(X,initialMixture,initialPost)
mixtureEM, postEM, ll = naive_em.run(X,initialMixture,initialPost)
common.plot(X,mixtureK, postK, "Kmean")
common.plot(X,mixtureEM, postEM, "EM Algo")

initialMixture, initialPost = common.init(X,2,0)
mixtureK, postK, distorision = kmeans.run(X,initialMixture,initialPost)
mixtureEM, postEM, ll = naive_em.run(X,initialMixture,initialPost)
common.plot(X,mixtureK, postK, "Kmean")
common.plot(X,mixtureEM, postEM, "EM Algo")

initialMixture, initialPost = common.init(X,3,0)
mixtureK, postK, distorision = kmeans.run(X,initialMixture,initialPost)
mixtureEM, postEM, ll = naive_em.run(X,initialMixture,initialPost)
common.plot(X,mixtureK, postK, "Kmean")
common.plot(X,mixtureEM, postEM, "EM Algo")

initialMixture, initialPost = common.init(X,4,0)
mixtureK, postK, distorision = kmeans.run(X,initialMixture,initialPost)
mixtureEM, postEM, ll = naive_em.run(X,initialMixture,initialPost)
common.plot(X,mixtureK, postK, "Kmean")
common.plot(X,mixtureEM, postEM, "EM Algo")