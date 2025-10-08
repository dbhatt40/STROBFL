#!/bin/bash


python dist_train_w_attack.py --dataset=fMNIST --k=20 --gar=kernel --C=1.0 --T=20 --iid=0.4 --mal --num_mal=2 --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self 

python dist_train_w_attack.py --dataset=fMNIST --k=40 --gar=kernel --C=0.5 --T=20 --iid=0.4 --mal --num_mal=4 --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self 

python dist_train_w_attack.py --dataset=fMNIST --k=60 --gar=kernel --C=0.3 --T=20 --iid=0.4 --mal --num_mal=6 --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self 

python dist_train_w_attack.py --dataset=fMNIST --k=80 --gar=kernel --C=0.25 --T=20 --iid=0.4 --mal --num_mal=8 --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self 

python dist_train_w_attack.py --dataset=fMNIST --k=100 --gar=kernel --C=0.20 --T=20 --iid=0.4 --mal --num_mal=10 --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self 
