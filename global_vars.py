#########################
# Purpose: Sets up global variables to be used throughout
########################

import argparse
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import logging
tf.get_logger().setLevel(logging.ERROR)

global data_dir


def dir_name_fn(args):
    # Setting directory name to store computed weights
    dir_name = 'weights/%s/%s/k%s_C%s_B%s' % (
        args.dataset, args.optimizer, args.k, args.C,args.B)
    # dir_name = 'weights/k{}_E{}_B{}_C{%e}_lr{}'
    output_file_name = 'output'

    output_dir_name = 'output_files/%s/%s-%s/N%d_M%s_A%.1f_I%.1f' % (
        args.dataset, args.optimizer, args.gar, args.ndrift,args.dmode,args.arate, args.ifactor)

    figures_dir_name = 'figures/%s/%s/k%s_C%s_B%s' % (
        args.dataset, args.optimizer, args.k, args.C, args.B)

    interpret_figs_dir_name = 'interpret_figs/%s/%s/k%s_C%s_B%s' % (
        args.dataset, args.optimizer, args.k, args.C, args.B)
    

    current_dir = os.getcwd()
		# Go up one level and into another directory (e.g., "data")
    data_dir = current_dir + "/data"
	
	
		
    if args.gar != 'avg':
        dir_name = dir_name + '_' + args.gar
        output_file_name = output_file_name + '_' + args.gar
        output_dir_name = output_dir_name + '_' + args.gar
        figures_dir_name = figures_dir_name + '_' + args.gar
        interpret_figs_dir_name = interpret_figs_dir_name + '_' + args.gar


    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    if not os.path.exists(figures_dir_name):
        os.makedirs(figures_dir_name)

    if not os.path.exists(interpret_figs_dir_name):
        os.makedirs(interpret_figs_dir_name)

    dir_name += '/'
    output_dir_name += '/'
    figures_dir_name += '/'
    interpret_figs_dir_name += '/'

    # print(dir_name)
    # print(output_file_name)

    return dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name, data_dir


def init():
    # Reading in arguments for the run
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='uci-sensor',
                        help="dataset to be used")
    parser.add_argument("--optimizer", default='adam',
                        help="optimizer to be used")
    parser.add_argument("--k", type=int, default=4, help="number of agents")
    parser.add_argument("--C", type=float, default=0.5,
                        help="fraction of agents per time step")
    parser.add_argument("--T", type=int, default=40, help="max time_steps")
    parser.add_argument("--B", type=int, default=25, help="agent batch size")
    parser.add_argument("--gar", type=str, default='avg', help="server aggregation rule")
    
    parser.add_argument("--ndrift", type=int, default=0, help="number drifted clients")
    parser.add_argument("--dmode", type=str, default='shared', help="type of drift - shared/independent")
    parser.add_argument("--ifactor", type=float, default=0.3, help="imbalance factor")
    parser.add_argument("--arate", type=float, default=1.0, help="imbalance factor")


    parser.add_argument("--steps", type=int, default=None,
                        help="GD steps per agent")
    parser.add_argument("--E", type=int, default=1,
                        help="epochs for each agent")
    parser.add_argument('--iid', type=float, default=1.0,
                        help="degree to which data is independent, identically distributed (range [0,1], higher is more iid)")
	
    parser.add_argument("--eta", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--train", default=True, action='store_true')
    parser.add_argument("--lr_reduce", action='store_true')
    parser.add_argument("--mal", default=False, action='store_true')
    parser.add_argument("--num_mal", type=int, default=0, help="number of malicious agents")
    parser.add_argument("--mal_obj", default='single',
                        help='Objective for malicious agent')
    parser.add_argument("--mal_strat", default='asyncFL',
                        help='Strategy for malicious agent')
    parser.add_argument("--mal_num", type=int, default=0,
                        help='Objective for simultaneous targeting')
    parser.add_argument("--mal_delay", type=int, default=0,
                        help='Delay for wait till converge')
    parser.add_argument("--mal_boost", type=float, default=10,
                        help='Boosting factor for alternating minimization attack')
    parser.add_argument("--mal_E", type=float, default=5,
                        help='Benign training epochs for malicious agent')
    parser.add_argument("--ls", type=int, default=1,
                        help='Training steps for each malicious step')
    parser.add_argument("--rho", type=float, default=1e-4,
                        help='Weighting factor for distance constraints')
    parser.add_argument("--data_rep", type=float, default=10,
                        help='Data repetitions for data poisoning')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help='GPUs to run on')
  

    global args
    args = parser.parse_args()
    # print(args)

    # making sure single agent run is only for the benign case
    #if args.k==1:
    #    assert args.mal==False


    # Moving rate of 1.0 leads to full overwrite
    global moving_rate

    global gpu_ids
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = [0]
    global num_gpus
    num_gpus = len(gpu_ids)

    global max_agents_per_gpu
    max_agents_per_gpu = 2



    global NUM_CLASSES, BATCH_SIZE, WINDOW_SIZE, DATA_DIM, NUM_DRIFTED 

    BATCH_SIZE = None
    NUM_CLASSES = None
    WINDOW_SIZE = None
    DATA_DIM = None
    NUM_DRIFTED = None
    BATCH_SIZE = args.B
    
    global T,C, k, gar
    global ndrift, dmode, arate, ifactor, B
    T = None
    C = None
    k = None
    gar = None
    
    T = args.T
    C = args.C
    k = args.k
    gar = args.gar
    ndrift = 0
    dmode = 'Shared'
    arate = 1.0
    ifactor = 0.3
    ndrift = args.ndrift
    dmode = args.dmode
    arate = args.arate
    ifactor = args.ifactor
    B = args.B

    global max_acc

    if args.dataset == 'air-quality':     
        DATA_DIM = 13
        BATCH_SIZE = 50
        WINDOW_SIZE = 50
        NUM_CLASSES = 1
        max_acc = 85.0
        max_agents_per_gpu = 2
        mem_frac = 0.05
        moving_rate = 1.0    
    elif args.dataset == 'synthetic-class1':     
       DATA_DIM = 2
       NUM_CLASSES = 4
       WINDOW_SIZE = 500
       NUM_DRIFTED = 0
       max_acc = 85.0
       max_agents_per_gpu = 2
       mem_frac = 0.05
       moving_rate = 1.0    
    if max_agents_per_gpu < 1:
        max_agents_per_gpu = 1

    global gpu_options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)

    global dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name

    dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name, data_dir = dir_name_fn(
        args)

    return args
