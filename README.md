# Sequential Mechanisms for Two-Phase Reviews

This is the Python implementation of the simulation results of the paper under the same title. 

# Data.py

This file contains functions of preparing data for testing the mechanism
Details are listed as follows.
| Item | Input  |	Output | 功能 |
|:--------:|:--------:|:--------:|:--------:|
| extract_number | string | score in that string | fetch the score in that line |
| papers | -  | - | a list stores all the papers' review info | 
| papers.forum | papers[id]['forum'] | string of forum name | - |  
| papers.authorids | papers[id]['authorids'] | string of authorids| get authors of the id-th paper |
| papers.decision | papers[id]['decision'] | Accepted or Rejected | get whether the id-th paper is accepted |
| papers.scores | papers[id]['scores'] | scores of different reviews | get the list of the id-th paper's score |



# two_phase.py

This contains the main implementation of the two-phase mechanism


| Item | Input  |	Output | 功能/注释   |
|:--------:|:--------|:--------|:--------|
| `TwoPhaseParams`| `n`:Just for testing，每个作者的文章数量，随便填，因为真实的ni不同，实验时未用到 <br> `m`,`m1`,`mu_q`, `sig_q`,`t`, `t1`,` t2`, `t_acc`, `minScore = 1`, `maxScore = 10` 是什么都很明显x | N/A | 存参数的类|
| `Compute_s_prior_prob`| `para`:`TwoPhaseParams`的实例<br> `integSigmaFactor = 5`:因为s的先验是无穷积分，只能近似在`mu_q`± `integSigmaFactor`* `sig_q`的范围算|`prior_s_prob`：字典，`prior_s_prob[(1,2,3)]` = 取到(1,2,3)的概率 |算长度为`para.m`的所有独特s的先验概率|
| `Expected_quality_of_combinations`| `para`:`TwoPhaseParams`的实例<br> `prior_s_prob`: s的先验 <br>`integSigmaFactor = 5`:同上|`expected_quality`:字典<br>`expected_quality[1,2,3]`= 得分(1，2，3)时质量的期望 |算长度为`para.m`的所有s独特组合的期望质量|
| `count_permutations`| `t`:数字组合，比如（1，2，1）|`total_permutations`:独特的排列数量，`count_permutations[(1,2,1)]`=3 |算重复的排列数，unused|
| `two_phase`|`full_q`:这个作者的所有文章的质量q<br>`para`：`TwoPhaseParams`的实例<br>`s_samples`：对这个作者的文章的分数采样，`len(s_samples)`=`sample_times`<br>`expected_quality`:字典，`Expected_quality_of_combinations`的输出, 长度为m<br> `expected_quality_2`:字典同上，但长度为m1<br>|`p1outcome_of_samples`：`p1outcome_of_samples[i][j]`第i组的第j篇是否进入phase2 <br>`p2outcome_of_samples`:`p2outcome_of_samples[i][j]`第i组的第j篇是否被接受<br> `review_times`：一共被review的次数<br> `accepted_q`：二维数组，该作者的所有sample被接受的所有文章质量<br> | 最主要的部分，模拟phase1 和phase2 |
| `sample_s` | `para`:`TwoPhaseParams`的实例</br>`q`:一个作者的所有文章质量q</br>`sampletimes`:采样次数 |`s_samples`: `s_samples[j][i]`是第j次抽样的第i个paper分数|这里每次抽样对每篇文章采m1个分数，phase1只用前m个|
| `test`|无，简陋的测试用 |无，建议print | 直接call就可以简单测试，这里简陋的假设每个n_i是一样的 |

# Experiment
This contains all the experiment

`ex.py` 调用`two_phase`，根据`t1`,`t2`,`tacc`的范围反复多次实验并将结果保存。运行：python ex.py m m1。

`sample_q.txt` 从均值5.5，方差1.0的正态分布中采样得到1000名作者的论文质量，每个作者的论文数量符合学来的分布。

`analysis.ipynb`: 根据实验结果画图等


</br>

# README from the original repo and author

## Eliciting Honest Information From Authors Using Sequential

This is the Python implementation of the simulation results of the paper under the same title. 

### Guide of Usage
First, to run our codes, packages including NumPy, SciPy, Random, and Matplotlib are required for plotting. All of the experiments are implemented on JuPyter Notebook version 6.1.4. with Anaconda Navigator 1.10.0 and Python 3.8.5.

The roles of the Python files are:
* "**Gaussian_model.ipynb**" includes the code for generating the simulation results for the Gaussian review noise model discussed in Section 5.2;
* "**Real-data.ipynb**" contains the code for estimating the Softmax review noise model with the ICLR datasets and for generating the simulation results in Section 5.3.

The "**Numerical_results**" folder saves some intermediate results that may take hours to run. For those experiments, users could skip the codes that are used to generate these results.
