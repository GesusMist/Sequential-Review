import two_phase as tp
import numpy as np
import time
import sys

def test(para, q, expected_quality, expected_quality_2):
    print("start: ",time.time())
    p1outcome_of_samples = []
    p2outcome_of_samples = []
    accepted_q = []
    review_burden = 0
    sample_times = 5000
    author_num = 1000
    total_num = 0

    q_all = np.zeros(sample_times)

    # 依次对每个作者进行two_phase
    for i in range(author_num):
        # 生成评论分数
        s_samples = tp.sample_s(para, q[i], sample_times)
        # 调用two_phase方法
        p1outcome_of_samples_i, p2outcome_of_samples_i, review_burden_i, accepted_q_i = tp.two_phase(q[i], para, s_samples, expected_quality, expected_quality_2)
        p1outcome_of_samples.append(p1outcome_of_samples_i)
        p2outcome_of_samples.append(p2outcome_of_samples_i)
        # 统计一些数据
        for j in range(len(accepted_q_i)):
            q_all[j] += np.sum(np.array(accepted_q_i[j])/1000)
            total_num += len(accepted_q_i[j])

        review_burden += review_burden_i
    print(len(q_all))
    avg = np.average(q_all)
    sig = 0.0
    for i in q_all:
        sig += (avg - i) ** 2
    sig /= sample_times
    print("review_burden: ",review_burden/sample_times)
    print("average_q: ", avg)
    print("total_num: ", total_num/sample_times)
    print("sig_q: ", sig)
    print(time.time())

    return [review_burden/sample_times, np.sum(q_all), total_num/sample_times, sig]

# m和m1不变，threshold在一定范围内变化
def experiment(m, m1):
    # 每次实验用同样的数据
    q = []
    with open('sample_q.txt', 'r') as file:
        i = file.readlines()
        for j in i:
            k = j.strip().split(' ')
            q.append(np.array(list(map(float, k))))
    #生成p1的期望质量字典
    para = tp.TwoPhaseParams(10, m, m1, 5.5, 1.0,-0.513, 5.0, 5.0, 5.0, 1, 10)
    prior_s_prob = tp.Compute_s_prior_prob(para)
    expected_quality = tp.Expected_quality_of_combinations(para, prior_s_prob)
    # 这里复制一份para改m=m1，用于生成p2的期望质量字典
    para_2 = tp.TwoPhaseParams(10, m1, m1, 5.5, 1.0,-0.513, 5.0, 5.0, 5.0, 1, 10)
    prior_s_prob_2 = tp.Compute_s_prior_prob(para_2)
    expected_quality_2 = tp.Expected_quality_of_combinations(para_2, prior_s_prob_2)
    # threshold的范围
    res = []
    for t1 in np.arange(3.0, 8.1, 0.5):
        for t2 in np.arange(3.0, 8.1, 0.5):
            for tacc in np.arange(4.5, 6.6, 1.0):
                # 调用单次实验的方法
                para = tp.TwoPhaseParams(10, m, m1, 5.5, 1.0,-0.513, t1, t2, tacc, 1, 10)
                print(t1, " ", t2, " ", tacc)
                # res[i] = 第i组threshold的结果，[burden，所有采样被接受quality之和，平均一次录取的论文数量，论文质量的方差，[threshold]]
                res.append([test(para, q, expected_quality, expected_quality_2), [t1, t2, tacc, 0]])
    np.save("result_"+str(m)+str(m1)+"_new1.npy", res)

print(sys.argv)

experiment(int(sys.argv[1]), int(sys.argv[2]))