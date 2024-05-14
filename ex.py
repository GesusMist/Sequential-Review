import two_phase as tp
import numpy as np
import time
import sys

def test(para, q, expected_quality):
    print("start: ",time.time())
    p1outcome_of_samples = []
    p2outcome_of_samples = []
    accepted_q = []
    review_burden = 0
    sample_times = 50
    author_num = 1000
    total_num = 0

    for i in range(author_num):
        #print("author",i," : ",time.time())
        s_samples = tp.sample_s(para, q[i], sample_times)
        #print(s_samples_sp[0][0])
        #print("author",i," sampled: ",time.time())
        p1outcome_of_samples_i, p2outcome_of_samples_i, review_burden_i, accepted_q_i = tp.two_phase(q[i], para, s_samples, expected_quality)
        p1outcome_of_samples.append(p1outcome_of_samples_i)
        p2outcome_of_samples.append(p2outcome_of_samples_i)
        accepted_q += accepted_q_i
        review_burden += review_burden_i
        total_num += len(accepted_q_i)
        
    # print(np.average(p1outcome_of_samples, axis = 1))
    # print(np.average(p2outcome_of_samples, axis = 1))
    print("review_burden: ",review_burden/sample_times/author_num)
    print("average_q: ",np.average(np.average(accepted_q, axis = 0)))
    print("total_num: ", total_num/sample_times)
    print(time.time())

    return [review_burden/sample_times/author_num, np.average(np.average(accepted_q, axis = 0)), total_num/sample_times]

def experiment(m, m1):
    q = []
    with open('sample_q.txt', 'r') as file:
        i = file.readlines()
        for j in i:
            k = j.strip().split(' ')
            q.append(np.array(list(map(float, k))))

    para = tp.TwoPhaseParams(10, m, m1, 5.5, 1.0,-0.513, 5.0, 5.0, 5.0, 1, 10)
    prior_s_prob = tp.Compute_s_prior_prob(para)
    expected_quality = tp.Expected_quality_of_combinations(para, prior_s_prob)
    
    res = []
    for t1 in np.arange(4.0, 7.0, 0.4):
        for t2 in np.arange(4.0, 7.0, 0.4):
            for tacc in np.arange(4.5, 6.6, 1.0):
                para = tp.TwoPhaseParams(10, m, m1, 5.5, 1.0,-0.513, t1, t2, tacc, 1, 10)
                print(t1, " ", t2, " ", tacc)
                res.append([test(para, q, expected_quality), [t1, t2, tacc]])
    np.save("result_"+str(m)+str(m1)+".npy", res)

print(sys.argv)

experiment(int(sys.argv[1]), int(sys.argv[2]))