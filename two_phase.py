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
    expected_quality = {}

    for combination in itertools.combinations_with_replacement(range(minScore, maxScore+1), m):

        def func(x): # q * P(s|q)P(q)/P(s)
            return x*                                                                                   \
                    norm.pdf(x,mu_q, sig_q) *                                                           \
                    np.prod(np.array([np.exp(t*(k - x) ** 2) /                                          \
                    (np.sum(np.exp(t * (scores[:] - x) ** 2) , axis = 0)) for k in combination]))/      \
                    prior_s_prob[tuple(combination)]
        expect_quality = integrate.quad(func, mu_q - integSigmaFactor * sig_q, mu_q + integSigmaFactor * sig_q)
        expected_quality[tuple(combination)] = expect_quality[0]

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


def sample_s(para,m, q, sampletimes = 10000):
    t = para.t
    # q = np.array([full_q[i] for i in range(math.ceil(len(full_q)/2))])
    q_reshape = q[np.newaxis, :]
    combinations = []
    prob_for_sampling = {}
    scores = np.arange(para.minScore, para.maxScore+1)
    for combination in itertools.combinations_with_replacement(range(para.minScore, para.maxScore+1), m):
        combinations.append(tuple(combination))
        prob_of_combination = np.exp(t*(np.array(combination)[:, np.newaxis] - q_reshape)**2) \
                            / np.sum(np.exp(t*(scores[:, np.newaxis] - q_reshape)**2), axis = 0)#算ombination中每个s的概率
        prob_for_sampling[tuple(combination)] = np.prod(prob_of_combination, axis = 0) * count_permutations(combination) #算combination的概率，乘组合数

    s_samples = np.zeros((len(q),sampletimes, m))
    for i in range(len(q)):
        number_samples = np.random.choice(len(combinations), sampletimes, p = [j[i] for j in list(prob_for_sampling.values())])
        #这里因为np.random.choice只能抽一维的,所以只能先抽index，再找到对应的dict

        s_samples[i] =  np.array([combinations[s] for s in number_samples])

    s_samples = np.transpose(s_samples, (1, 0, 2))
    #维度换一下，现在s_samples[j][i]是第j次抽样的第i个paper
    return s_samples

def phase1_into_phase2_prob(full_q, para,s_samples, expected_quality, sampletimes = 100000):

    # t = para.t
    t1 = para.t1
    t2 = para.t2
    # m = para.m
    # scores = np.arange(para.minScore, para.maxScore+1)

    # q = np.array([full_q[i] for i in range(math.ceil(len(full_q)/2))]) # here q is the first half of the full_q
    # q_reshape = q[np.newaxis, :]
    # prob_for_sampling = {}  #采样每一个组合的概率
    prob_per_combination ={}
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
    outcome_of_samples = np.zeros((len(s_samples), len(full_q)))  # stores the result of phase 1. outcome_of_samples[j][i] = 1 means jth sample's ith paper is into phase 2.
    

    for index, sample in enumerate(s_samples):                  # indexth sample
        outcome_of_a_sample_set = np.zeros(len(full_q))         # stores the result of phase 1 of a set. outcome_of_a_sample_set[i] = 1 means ith paper is into phase 2.
        ifall = 1                                              # 是否前i个paper都进了phase2
        for i, s in enumerate(sample):  # ith paper in the sample


            if tuple(s) not in prob_per_combination.keys():
                prob_per_combination[tuple(s)] = np.zeros(math.ceil(len(full_q)/2))#如果不存在先init


            if expected_quality[tuple(s)] > t1:  
                outcome_of_a_sample_set[i] = 1                      #如果这个s的质量大于t1，那么这个paper就进入phase2    
                
                prob_per_combination[tuple(s)][i] +=1/sampletimes   #q[i]的这个combination的概率加1/sampletimes

            elif expected_quality[tuple(s)] > t2 and ifall:
                outcome_of_a_sample_set[i] = 1
                
                prob_per_combination[tuple(s)][i] +=1/sampletimes

            else:
                outcome_of_a_sample_set[i] = 0
                ifall = 0


        if ifall:
            outcome_of_a_sample_set = np.ones(len(full_q))          #如果前ceil(n/2)paper进入phase2，那么这个sample就全进入phase2
        else:
            for i in range(len(sample), len(full_q)):
                outcome_of_a_sample_set[i] = 0
        outcome_of_samples[index] = outcome_of_a_sample_set
        # print(outcome_of_a_sample_set)

    prob_into_phase2 = np.sum(outcome_of_samples, axis = 0)/sampletimes #概率 = 频数/总次数
    return prob_into_phase2 , prob_per_combination

def phase2(para, q, prob_per_combination_phase1):
    t_acc = para.t_acc
    t = para.t
    m1 = para.m1
    m = para.m
    minScore = para.minScore
    maxScore = para.maxScore

    scores = np.arange(minScore, maxScore+1) 

    

    # for the first half
    q1 = np.array([q[i] for i in range(math.ceil(len(q)/2))])
    q1_reshape = q1[np.newaxis, :]
    q1_acc_probability = np.zeros(len(q1))

    for combination in itertools.combinations_with_replacement(range(minScore, maxScore), m1-m):
        for p1_combination in itertools.combinations_with_replacement(range(minScore, maxScore), m):
            if np.sum(combination)+np.sum(p1_combination) >= t_acc * m1:
                if tuple(p1_combination) not in prob_per_combination_phase1.keys():
                    prob_per_combination_phase1[tuple(p1_combination)] = np.zeros(math.ceil(len(q)/2))
                prob = np.exp(t*(np.array(combination)[:, np.newaxis] - q1_reshape)**2) / np.sum(np.exp(t*(scores[:, np.newaxis] - q1_reshape)**2), axis = 0) 
                temp = np.prod(prob, axis = 0)* prob_per_combination_phase1[tuple(p1_combination)]*count_permutations(tuple(combination))
                q1_acc_probability += temp
                # print("p1:",p1_combination," ","c:",combination," ",temp)

    # print(q1_acc_probability)
    # for the second half
    q2 = np.array([q[i] for i in range(math.ceil(len(q)/2), len(q))])
    q2_reshape = q2[np.newaxis, :]
    q2_acc_probability = np.zeros(len(q2))
    for combination in itertools.combinations_with_replacement(range(minScore, maxScore), m1):
        if np.sum(combination) >= t_acc * m1:
            prob = np.exp(t*(np.array(combination)[:, np.newaxis] - q2_reshape)**2) / np.sum(np.exp(t*(scores[:, np.newaxis] - q2_reshape)**2), axis = 0) 
            q2_acc_probability += np.prod(prob, axis = 0)*count_permutations(combination)


    q_acc_probability = np.concatenate((q1_acc_probability, q2_acc_probability), axis = 0)

    return q_acc_probability

def test():
    q = np.array([9,8,7,6,5,4,3,2,1,0])
    para = TwoPhaseParams(10, 3, 5, 5.5, 1.5,-0.513, 5.5, 5.0, 5.6, 1, 10)
    q = sample_q(para, 20)
    print(q.shape)
    prior_s_prob = Compute_s_prior_prob(para)
    expected_quality = Expected_quality_of_combinations(para, prior_s_prob)
    prob_all=np.zeros((20,10))
    prob_per_combination = np.empty(20, dtype=dict)
    acc_prob = np.zeros((20,10))
    all_s_samples=np.zeros(20,dtype=np.ndarray)
    for i in range(20):
        s_samples = sample_s(para, para.m, q[i][0:math.ceil(para.n/2)], 100000)
        all_s_samples[i] = s_samples
        prob_all[i] , prob_per_combination[i] = phase1_into_phase2_prob(q[i], para,s_samples, expected_quality)
        print(prob_all[i])
        acc_prob[i] = phase2(para, q[i], prob_per_combination[i])
    print(np.average(acc_prob, axis = 0))

    # for key in prob_per_combination:
    #     print("key:",key," ",prob_per_combination[key])
    # acc_prob = phase2(para, q, prob_per_combination)
    # print(acc_prob)
    

test()

# print( itertools.combinations_with_replacement(range(1, 5),2)[1])