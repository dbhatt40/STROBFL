# Model Poisoning Attacks

This code accompanies the paper 'Analyzing Federated Learning through an Adversarial Lens' which has been accepted at ICML 2019. It assumes that the Fashion MNIST data and Census data have been downloaded to /home/data/ on the user's machine.

Dependencies: Tensorflow-1.8, keras, numpy, scipy, scikit-learn

To run federated training with 10 agents and standard averaging based aggregation, use
```
python dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --gar=avg
```
To run the basic targeted model poisoning attack, use
```
python dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge --gar=avg
```

To run the alternating minimization attack with distance constraints with the parameters used in the paper, run
```
python dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self --rho=1e-4 --gar=avg --ls=10 --mal_E=10

python dist_train_w_attack.py --dataset=fMNIST --E=1 --T=20 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self --rho=1e-4 --gar=pca    --ls=10 --mal_E=10 --k=20 --C=0.5 --num_mal=2 --iid=1.0
```
The function of the various parameters that are set by `utils/globals_vars.py` is given below.

| Parameter   | Function                                               |
|-------------|--------------------------------------------------------|
| --gar       | Gradient Aggregation Rule                              |
| --eta       | Learning Rate                                          |
| --k         | Number of agents                                       |
| --C         | Fraction of agents chosen per time step                |
| --E         | Number of epochs for each agent                        |
| --T         | Total number of iterations                             |
| --B         | Batch size at each agent                               |
| --mal_obj   | Single or multiple targets                             |
| --mal_num   | Number of targets                                      |
| --mal_strat | Strategy to follow                                     |
| --mal_boost | Boosting factor                                        |
| --mal_E     | Number of epochs for malicious agent                   |
| --ls        | Ratio of benign to malicious steps in alt. min. attack |
| --rho       | Weighting factor for distance constraint               |
| --num_mal   | Number of malicious clients                            |
| --iid       | Extent to which data is iid between clients            |

The other attacks can be found in the file `malicious_agent.py`.

## Logging in to the gpel machines
In order to connect to the GPU machines, you must be on the OU network either in person or via VPN. Instructions for setting up VPN access can be found here: https://www.ou.edu/marcomm/cms/get-started/vpn

Once on the OU network, you can connect to a machine via ssh (PuTTY on windows).

Connection information:

Host Name: gpel[number].cs.nor.ou.edu

Port: 22

Connection Type: SSH

Replace [number] with a number between 8 and 13 inclusive to choose which of the 6 gpel machines to use. I have always used gpel12, and I know that the current settings work on that machine. I'm not sure whether the others have different GPUs or not.

One you connect, log in with your OU 4x4 and password.

## Graduate student lab machine
There is another GPU machine in the graduate student computer lab. This computer's GPU has more memory than the gpel machines, but it has a CUDA compute capability that is too low for the latest versions of TensorFlow. This caused all kinds of problems, and I was unable to get it to work. You could try to get it to work to take advantage of the extra memory and speed up training if you want. You'll have to get someone to set up an account for you on this machine. Egawati Panjei helped me via Dr. Gruenwald.

Connection information:

Host Name: [username]@iverson.cs.nor.ou.edu

Port: 22

Connection Type: SSH

## Setting up the environment
Once logged into a gpel machine install miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

Clone your fork of this repo to your local machine. Move the folder data/ one directory up into the directory /home/[4x4]/. Enter the data directory and download the files for the fMNIST dataset into it: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion. Navigate back into the ModelPoisoning Folder.

Then recreate the conda environment with this command:

```
conda env create -f environment.yml
```

Activate the conda environment with
```





```

## Running Experiments
The shell commands for collecting one repetition of data are in DataCollection.sh. Running this script takes ~1 day. You begin running the script in the background and log out by using the following commands:
```
nohup bash DataCollection.sh &
exit
```
Standard out will be redirected to the file nohup.out. The results (records of performance metrics) will be created in the folder output_files/. The R script for generating the visualizations is in the folder results/. The csvs for that script were created by copying the relevant data points from the files in output_files/.

## Configuration for optimal GPU usage
The GPUs we have access to have a very limited amount of memory. If we provision too much memory, the program crashes. But if we provision too little, the experiments can take unnecessarily long to run. To examine the amount of memory used as you run an experiment, connect with a second ssh/PuTTY window and use the command
```
watch -n 1 nvidia-smi
```
Each agent being trained will appear as a separate thread/process, and you can monitor the total memory usage and the usage of each process.

If your runs are crashing from OOM or you have too much unused memory, there are several parameters you can adjust to fix this:

* Batch size: in global_vars.py adjust the BATCH_SIZE variable for the dataset you are using and adjust the --B flag (see the above table) to match. Reducing the batch size reduces the per agent memory demand.
* max_agents_per_gpu: in global_vars.py adjust the max_agents_per_gpu variable for the dataset you are using. Reducing this reduces the memory demand by training fewer agents on the gpu at one time.

## Reading list
The following are the most important papers that I have encountered to understand this research:

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017, April). Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics (pp. 1273-1282). PMLR.

This introduces the idea of federated learning.

2.  Bhagoji, A. N., Chakraborty, S., Mittal, P., & Calo, S. (2019, May). Analyzing federated learning through an adversarial lens. In International Conference on Machine Learning (pp. 634-643). PMLR.

This introduces the model poisoning attack. This is the paper this code was originally created for at https://github.com/inspire-group/ModelPoisoning

3. Tolpegin, V., Truex, S., Gursoy, M. E., & Liu, L. (2020, September). Data poisoning attacks against federated learning systems. In European Symposium on Research in Computer Security (pp. 480-501). Springer, Cham.

Introduces the PCA defense

4. Li, D., Wong, W. E., Wang, W., Yao, Y., & Chau, M. (2021, August). Detection and mitigation of label-flipping attacks in federated learning systems with KPCA and K-means. In 2021 8th International Conference on Dependable Systems and Their Applications (DSA) (pp. 551-559). IEEE.

Introduced the kPCA defense

5. Awan, S., Luo, B., & Li, F. (2021, October). Contra: Defending against poisoning attacks in federated learning. In European Symposium on Research in Computer Security (pp. 455-475). Springer, Cham.

Introduces the CONTRA defense

## Next Steps

* Review the implementation of the algorithms in dist_train_w_attack.py. Some specific points:
    * Review the loops in the CONTRA implementation. Are they referring to all of the clients or only those included in that communication round? Change the implementation if you interpret this differently than I did.
    * Figure out what values to use for the parameters Delta and the similarity threshold (I chose 0.9) in the CONTRA algorithm. The paper mentions suggestions for Delta based upon which classification task they were doing, but I couldn't find what values they used for the threshold.
    * The PCA and kPCA algorithms mention only including that which is connected to the output layer in the dimensionality reduction. This implementation doens't do this right now, but maybe it should.
    * Look into the KernelPCA documentation and decide which settings to use.
* Important: the visualizations currently only use data from one run of the experiments. We want to do multiple runs and average the results, but I rand out of time.
* Decide how you want to parametrize iid (data is distributed in the main method in dist_train_w_attack.py). The way I did it doesn't seem great; currently, as the parameter increases the data becomes iid very quickly. Instead, maybe measure it by the number of data slices per client and look at small numbers, i.e. 1 to 10 slices per client. This could be more meaningful.
* Potential future direction: see if the results are different with different datasets (census, CIFAR-10). We only use fMNIST right now.
* Potential future direction: additional defense methods?
* Potential future direction: implement a new defense method?
