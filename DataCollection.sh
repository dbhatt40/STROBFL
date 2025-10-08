#!/bin/bash

python dist_train_w_attack.py --dataset=fMNIST --E=1 --T=20 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self --rho=1e-4 --gar=contra --ls=10 --mal_E=10 --k=20 --C=0.5 --num_mal=2 --iid=0


