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
    sample_times = 5000
    author_num = 1000
    total_num = 0

    q_all = np.zeros(sample_times)

    for i in range(author_num):
        #print("author",i," : ",time.time())
        s_samples = tp.sample_s(para, q[i], sample_times)
        #print(s_samples_sp[0][0])
        #print("author",i," sampled: ",time.time())
        p1outcome_of_samples_i, p2outcome_of_samples_i, review_burden_i, accepted_q_i = tp.two_phase(q[i], para, s_samples, expected_quality)
        p1outcome_of_samples.append(p1outcome_of_samples_i)
        p2outcome_of_samples.append(p2outcome_of_samples_i)

        #print(accepted_q_i)
        for j in range(len(accepted_q_i)):
            q_all[j] += np.sum(np.array(accepted_q_i[j])/1000)
            total_num += len(accepted_q_i[j])

        review_burden += review_burden_i
    
    avg = np.average(q_all)
    sig = 0.0
    for i in q_all:
        sig += (avg - i) ** 2
    sig /= sample_times
        
    # print(np.average(p1outcome_of_samples, axis = 1))
    # print(np.average(p2outcome_of_samples, axis = 1))
    print("review_burden: ",review_burden/sample_times)
    print("average_q: ", avg)
    print("total_num: ", total_num/sample_times)
    print("sig_q: ", sig)
    print(time.time())

    return [review_burden/sample_times, np.sum(q_all), total_num/sample_times, sig]

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
    
    # s_samples = []
    # for i in range(1000):
    #     s_samples.append(tp.sample_s(para, q[i], 5000))
    
    res = []
    for t1 in np.arange(3.0, 8.1, 0.5):
        for t2 in np.arange(3.0, 8.1, 0.5):
            for tacc in np.arange(4.5, 6.6, 1.0):
                para = tp.TwoPhaseParams(10, m, m1, 5.5, 1.0,-0.513, t1, t2, tacc, 1, 10)
                print(t1, " ", t2, " ", tacc)
                res.append([test(para, q, expected_quality), [t1, t2, tacc, 0]])
    np.save("result_"+str(m)+str(m1)+"_new.npy", res)

print(sys.argv)

experiment(int(sys.argv[1]), int(sys.argv[2]))