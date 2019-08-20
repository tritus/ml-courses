import numpy as np
import kmeans
import common
import naive_em
import em

toy_X = np.loadtxt("toy_data.txt")

def test_seeds(X, K):
	print("\n############## KMEAN K=" + str(K) + " ###############")

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


def test_em_seeds(X, K):
	print("\n############## EM K=" + str(K) + " ###############")

	mixture0, post0 = common.init(X,K,0)
	mixture1, post1 = common.init(X,K,1)
	mixture2, post2 = common.init(X,K,2)
	mixture3, post3 = common.init(X,K,3)
	mixture4, post4 = common.init(X,K,4)

	cost0 = em.run(X,mixture0,post0)[2]
	cost1 = em.run(X,mixture1,post1)[2]
	cost2 = em.run(X,mixture2,post2)[2]
	cost3 = em.run(X,mixture3,post3)[2]
	cost4 = em.run(X,mixture4,post4)[2]

	print("K=" + str(K) + " seed=0 : likelihood=" + str(cost0))
	print("K=" + str(K) + " seed=1 : likelihood=" + str(cost1))
	print("K=" + str(K) + " seed=2 : likelihood=" + str(cost2))
	print("K=" + str(K) + " seed=3 : likelihood=" + str(cost3))
	print("K=" + str(K) + " seed=4 : likelihood=" + str(cost4))


# K mean initialization

test_seeds(toy_X, 1)
test_seeds(toy_X, 2)
test_seeds(toy_X, 3)
test_seeds(toy_X, 4)

# EM algo
print("############## EM Algorythme implemented ###############")
mixture, post = common.init(toy_X,3,0)
naive_em_estimate = naive_em.run(toy_X,mixture,post)[2]
print("naive EM log likelihood : " + str(naive_em_estimate))

print("############## Some Tests ######################")
initialMixture, initialPost = common.init(toy_X,1,0)
mixtureEM1, postEM1, ll1 = naive_em.run(toy_X,initialMixture,initialPost)

initialMixture, initialPost = common.init(toy_X,2,0)
mixtureEM2, postEM2, ll2 = naive_em.run(toy_X,initialMixture,initialPost)

initialMixture, initialPost = common.init(toy_X,3,0)
mixtureEM3, postEM3, ll3 = naive_em.run(toy_X,initialMixture,initialPost)

initialMixture, initialPost = common.init(toy_X,4,0)
mixtureEM4, postEM4, ll4 = naive_em.run(toy_X,initialMixture,initialPost)

print("BIC K1 : " + str(common.bic(toy_X, mixtureEM1, ll1)))
print("BIC K2 : " + str(common.bic(toy_X, mixtureEM2, ll2)))
print("BIC K3 : " + str(common.bic(toy_X, mixtureEM3, ll3)))
print("BIC K4 : " + str(common.bic(toy_X, mixtureEM4, ll4)))

X_netflix = np.loadtxt("netflix_incomplete.txt")
test_em_seeds(X_netflix, 1)
#test_em_seeds(X_netflix, 12)

X_gold = np.loadtxt('netflix_complete.txt')
mixture4, post4 = common.init(X_netflix,12,1)
mixture, post, cost4 = em.run(X_netflix,mixture4,post4)
X_pred = em.fill_matrix(X_netflix,mixture)

rmse_result = common.rmse(X_gold, X_pred)
print("RMSE between prediction and GOLD is : " + str(rmse_result))