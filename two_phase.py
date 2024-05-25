#import openreview
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.stats import norm
from collections import Counter
import collections
import itertools
from math import comb
import json
from datetime import datetime
import math
from math import factorial
from scipy import integrate
import time

class TwoPhaseParams:
    def __init__(self, n, m, m1, mu_q, sig_q,t, t1, t2, t_acc, minScore = 1, maxScore = 10 ):
        self.n = n
        self.m = m
        self.m1 = m1
        self.mu_q = mu_q
        self.sig_q = sig_q
        self.t = t
        self.t1 = t1
        self.t2 = t2
        self.t_acc = t_acc
        self.minScore = minScore
        self.maxScore = maxScore


def sample_q(para, nmc):
    samples = np.random.normal(loc=para.mu_q, scale=para.sig_q, size=(nmc,para.n))
    return np.sort(samples, axis = 1, kind='quicksort')[:,::-1]

# 采样N=1000个作者，每个作者的文章数量服从学来的分布
def sample_q_dis(para):
    paper_number_distribution = [0, 659, 177, 73, 27, 26, 10, 8, 20]
    res = []
    for i in range(1,9):
        for j in range(paper_number_distribution[i]):
            temp_sample = np.random.normal(para.mu_q, para.sig_q, i)
            res.append(np.sort(temp_sample)[::-1])
    return res


#def Compute_s_prior_prob(mu_q, sig_q, t, minScore, maxScore, m, integSigmaFactor = 5): 
def Compute_s_prior_prob(para, integSigmaFactor = 5): 
    # P(s) = ∫P(s|q)P(q)dq
    # Output: a dict with key = sorted distinct combination of numbers from minScore to maxScore with length m
    # Example: para.m = 3 and para.minScore = 1, para.maxScore = 10  =>  prior_s_prob[(1,2,10)]  is the prior probability of getting scores 1,2,10 

    mu_q = para.mu_q
    sig_q = para.sig_q
    t = para.t
    m = para.m
    minScore = para.minScore
    maxScore = para.maxScore

    scores = np.arange(minScore, maxScore+1) 
    sum = 0                #just for debug     
    prior_s_prob = {}
    for combination in itertools.combinations_with_replacement( range(minScore, maxScore+1),m):    
        def func(x):
            return norm.pdf(x, mu_q, sig_q)*                          \
                np.prod( np.array([np.exp(t * (k - x) ** 2)/          \
                (np.sum(np.exp(t * (scores[:] - x) ** 2) , axis = 0)) \
                for k in combination]))
            ## Bayes  P(q|s) = P(s|q)P(q)/P(s)
        
        prob = integrate.quad(func, mu_q - integSigmaFactor * sig_q, mu_q + integSigmaFactor * sig_q)


        prior_s_prob[tuple(combination)] = prob[0]
        sum+=prob[0]
    #print(sum) 
    return prior_s_prob


def Expected_quality_of_combinations( para , prior_s_prob, integSigmaFactor = 5):  
    #Compute the expected quality of all possible sorted distinct combinations of scores
    #output: a dict with key = sorted distinct combination of numbers from minScore to maxScore with length m
    #you'll get a dict, dict[(1,2,3)] is the expected quality of getting scores 1,2,3
    mu_q = para.mu_q
    sig_q = para.sig_q
    t = para.t
    m = para.m
    minScore = para.minScore
    maxScore = para.maxScore

    scores = np.arange(minScore, maxScore+1) 
    # expected_quality = {}
    expected_quality = np.zeros(100000)

    for combination in itertools.combinations_with_replacement(range(minScore, maxScore+1), m):
    # for combination in [tuple(i,j,k) for i in range(minScore, maxScore+1) for j in range(minScore, maxScore+1) for k in range(minScore, maxScore+1)]:
        def func(x): # q * P(s|q)P(q)/P(s)
            return x*                                                                                   \
                    norm.pdf(x,mu_q, sig_q) *                                                           \
                    np.prod(np.array([np.exp(t*(k - x) ** 2) /                                          \
                    (np.sum(np.exp(t * (scores[:] - x) ** 2) , axis = 0)) for k in combination]))/      \
                    prior_s_prob[tuple(combination)]
        expect_quality = integrate.quad(func, mu_q - integSigmaFactor * sig_q, mu_q + integSigmaFactor * sig_q)
        for i in set(itertools.permutations(combination)): #这里为了避免在two_phase的超大loop里使用排序，于是在expect_quality中存了所有排列 
            # expected_quality[i] = expect_quality[0]
            temp = 0
            for j in i:
                temp = temp * 10 + j - 1
            expected_quality[temp] = expect_quality[0]

    return expected_quality



def count_permutations(t):
    #Out put the number of distinct permutations of a list
    if len(t) == 1:
        return 1
    # Count occurrences of each unique element
    element_counts = Counter(t)
    # Calculate the factorial of the total number of items
    total_permutations = factorial(sum(element_counts.values()))
    # Divide by the factorial of each element count
    for count in element_counts.values():
        total_permutations //= factorial(count)
    return total_permutations


def sample_s(para, q, sampletimes = 10000):
    t = para.t
    # q = np.array([full_q[i] for i in range(math.ceil(len(full_q)/2))])
    q_reshape = q[np.newaxis, :]
    prob_for_sampling = {}
    scores = np.arange(para.minScore, para.maxScore+1)
    for i in range(para.minScore, para.maxScore+1):

        prob_of_i = np.exp(t*(np.array([i])[:, np.newaxis] - q_reshape)**2) \
                            / np.sum(np.exp(t*(scores[:, np.newaxis] - q_reshape)**2), axis = 0)#算i中每个s的概率
        prob_for_sampling[i] = prob_of_i

    s_samples = np.zeros((len(q),sampletimes*para.m1))


    for i in range(len(q)):
        s_samples[i] = np.random.choice(range(para.minScore, para.maxScore+1), sampletimes*para.m1, p = [j[0][i] for j in list(prob_for_sampling.values())])

    s_samples = s_samples.reshape(len(q),sampletimes, para.m1)
    s_samples = np.transpose(s_samples, (1, 0, 2))
    #维度换一下，现在s_samples[j][i]是第j次抽样的第i个paper
                
    return s_samples

def two_phase(full_q, para,s_samples, expected_quality, expected_quality_2):

    # t = para.t
    t1 = para.t1
    t2 = para.t2
    t_acc = para.t_acc
    m = para.m
    m1  = para.m1
    ###########################↓废弃代码
    # m = para.m
    # scores = np.arange(para.minScore, para.maxScore+1)

    # q = np.array([full_q[i] for i in range(math.ceil(len(full_q)/2))]) # here q is the first half of the full_q
    # q_reshape = q[np.newaxis, :]
    # prob_for_sampling = {}  #采样每一个组合的概率
    # prob_per_combination ={}
    # combinations = []
    # for combination in itertools.combinations_with_replacement(range(para.minScore, para.maxScore+1), m):
        # prob_of_combination = np.exp(t*(np.array(combination)[:, np.newaxis] - q_reshape)**2) \
                            # / np.sum(np.exp(t*(scores[:, np.newaxis] - q_reshape)**2), axis = 0)#算ombination中每个s的概率
        # prob_for_sampling[tuple(combination)] = np.prod(prob_of_combination, axis = 0) * count_permutations(combination) #算combination的概率，乘组合数
        # combinations.append(tuple(combination))     #所有的组合，之后用
        # prob_per_combination[tuple(combination)] = np.zeros(len(q))     #init每个combination的概率

    # s_samples = np.zeros((len(q),sampletimes, m)) # stores the samples of phase 1. s_samples[i][j] is the jth sample's ith paper.

    # print(np.sum(np.array(p)))
    # print(prob_for_sampling.values()[i])
    # for i in range(len(q)):
        # number_samples = np.random.choice(len(combinations), sampletimes, p = [j[i] for j in list(prob_for_sampling.values())])
                                    #这里因为np.random.choice只能抽一维的,所以只能先抽index，再找到对应的dict

        # s_samples[i] =  np.array([combinations[s] for s in number_samples])

 
    # s_samples = np.transpose(s_samples, (1, 0, 2))
    #维度换一下，现在s_samples[j][i]是第j次抽样的第i个paper
    # print(s_samples)
    # print(s_samples)

    #####################↑废弃



    review_times = 0
    accepted_q = []
    p1outcome_of_samples = np.zeros((len(s_samples), len(full_q)))  # stores the result of phase 1. outcome_of_samples[j][i] = 1 means jth sample's ith paper is into phase 2.
    p2outcome_of_samples = np.zeros((len(s_samples), len(full_q)))

    # for index, sample in enumerate(s_samples):  
    for i in range(len(s_samples)):                 # indexth sample
        p1outcome_of_a_sample_set = np.zeros(len(full_q))         # stores the result of phase 1 of a set. outcome_of_a_sample_set[i] = 1 means ith paper is into phase 2.
        p2outcome_of_a_sample_set = np.zeros(len(full_q))
        accepted_q_of_a_sample_set = []
        ifall = 1                                              # 是否前i个paper都进了phase2
        # for i, s in enumerate(sample):  
        for j in range(math.ceil(len(s_samples[i]/2))):        #对前n/2个paper

            temp = 0
            for tt in s_samples[i][j][0:m]:
                temp = temp * 10 + tt - 1
            temp = int(temp)

            temp2 = 0
            for tt in s_samples[i][j]:
                temp2 = temp2 * 10 + tt - 1
            temp2 = int(temp2)


            if expected_quality[temp] > t1:
            # if expected_quality[tuple(s_samples[i][j][0 : m])] > t1:   #如果这个s的质量大于t1，那么这个paper就进入phase2 
                p1outcome_of_a_sample_set[j] = 1                            #进入phase2
                if expected_quality_2[temp2] >= t_acc:                   # 平均分数 > t_acc 则接受     
                    p2outcome_of_a_sample_set[j] = 1                        # paper被接受
                    accepted_q_of_a_sample_set.append(full_q[j])
                
                review_times += m1

            elif expected_quality[temp] > t2 and ifall:                    
            #　elif expected_quality[tuple(s_samples[i][j][0 : m])] > t2 and ifall:
                p1outcome_of_a_sample_set[j] = 1
                if expected_quality_2[temp2] >= t_acc:                       
                    p2outcome_of_a_sample_set[j] = 1
                    accepted_q_of_a_sample_set.append(full_q[j])
                review_times += m1

            else:
                ifall = 0                                                  #没进phase2
                review_times += m


        if ifall:
            p1outcome_of_a_sample_set = np.ones(len(full_q))          #如果前ceil(n/2)paper进入phase2，那么这个sample就全进入phase2
            for j in range(math.ceil(len(s_samples[i]/2)), len(s_samples[i]) ): #review后n/2个paper
                if expected_quality_2[temp2] >= t_acc:   
                    p2outcome_of_a_sample_set[j] = 1
                    accepted_q_of_a_sample_set.append(full_q[j])
                review_times += m1
        
        accepted_q.append(accepted_q_of_a_sample_set)
        p1outcome_of_samples[i] = p1outcome_of_a_sample_set
        p2outcome_of_samples[i] = p2outcome_of_a_sample_set
        # print(outcome_of_a_sample_set)


    return p1outcome_of_samples, p2outcome_of_samples, review_times, accepted_q

# def phase2(full_q, para, s_samples, phase1_outcome):
#     t_acc = para.t_acc

#     m1 = para.m1




#     outcome_of_samples = np.zeros((len(s_samples), len(full_q)))
#     for index, sample in enumerate(s_samples):  
#         for i, s in enumerate(sample):
#             if phase1_outcome[index][i] == 1:
#                 if np.sum(s[0:m1]) > t_acc:
#                     outcome_of_samples[index][i] = 1
#                 else:
#                     outcome_of_samples[index][i] = 0
#             else:
#                 outcome_of_samples[index][i] = 0

#     return q_acc_probability

def test():
    q = np.array([9,8,7,6,5,4,3,2,1,0])
    para = TwoPhaseParams(10, 3, 5, 5.5, 1.5,-0.513, 5.5, 5.0, 5.6, 1, 10)
    q = sample_q_dis(para)
    #print(q.shape)

    print("start: ",time.time())
    prior_s_prob = Compute_s_prior_prob(para)
    print("compute prior: ",time.time())
    expected_quality = Expected_quality_of_combinations(para, prior_s_prob)
    print("compute expected_quality: ",time.time())
    p1outcome_of_samples = []
    p2outcome_of_samples = []
    accepted_q = []
    review_burden = 0
    sample_times = 10000
    author_num = 1000

    for i in range(author_num):
        print("author",i," : ",time.time())
        s_samples = sample_s(para, q[i], sample_times)
        print("author",i," sampled: ",time.time())
        p1outcome_of_samples_i, p2outcome_of_samples_i, review_burden_i, accepted_q_i = two_phase(q[i], para, s_samples, expected_quality)
        p1outcome_of_samples.append(p1outcome_of_samples_i)
        p2outcome_of_samples.append(p2outcome_of_samples_i)
        accepted_q += accepted_q_i
        review_burden += review_burden_i
        
    # print(np.average(p1outcome_of_samples, axis = 1))
    # print(np.average(p2outcome_of_samples, axis = 1))
    print("review_burden: ",review_burden/sample_times/author_num)
    print("average_q: ",np.average(np.average(accepted_q, axis = 0)))
    print(np.average(np.average(p1outcome_of_samples, axis = 1), axis = 0))
    print(np.average(np.average(p2outcome_of_samples, axis = 1), axis = 0))
    # for key in prob_per_combination:
    #     print("key:",key," ",prob_per_combination[key])
    # acc_prob = phase2(para, q, prob_per_combination)
    # print(acc_prob)
    
if __name__ == "__main__":
    test()


# print( itertools.combinations_with_replacement(range(1, 5),2)[1])