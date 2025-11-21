import argparse
import sys
import torch
import warnings
warnings.filterwarnings("ignore")
from configuration import config
from datasets import *
from methods.AACL import AACL

methods = {"aacl" : AACL}

def main(arg_list=None):
    if arg_list:
        sys.argv = [sys.argv[0]] + arg_list
    args = config.base_parser() 
    print(f"[main] Parsed Args:\n{args}")
    print("#################################################################################################################")
    trainer = methods[args.mode](**vars(args))
    trainer.run()


def main_runner():
    parser = argparse.ArgumentParser(description="Run  AACL continual learning experiments with configurable parameters.")

    # General configuration
    parser.add_argument("--mode", type=str, default="aacl", help="aacl")
    parser.add_argument("--eta" , default=2, help="default value :2")
    parser.add_argument("--beta" , default=0.8, help="default value :1")
    parser.add_argument("--dataset", type=str, default="cifar100", help="cifar100,tinyimagenet,imagenet-r,imagenet1000")
    parser.add_argument("--n_tasks", type=int, default=5,help="default value :5")
    parser.add_argument("--n", type=int, default=50,help="default value :50")
    parser.add_argument("--m", type=int, default=10,help="default value :10")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gpu_transform", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--note", type=str, default="AACL")
    parser.add_argument("--data_dir", type=str, default="local_datasets")
    parser.add_argument("--n_worker", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="vit", help="vit,resnet18,resnet34,resnet50,efficientnet_b0,vitB16")
    parser.add_argument("--opt_name", type=str, default="adam")
    parser.add_argument("--sched_name", type=str, default="default")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batchsize", type=int, default=64,help="default value :64")
    parser.add_argument("--memory_size", type=int, default=0)
    parser.add_argument("--online_iter", type=int, default=3)
    parser.add_argument("--eval_period", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=5000,help="default value :5000")
    parser.add_argument("--gamma_main", type=float, default=100)
    parser.add_argument("--mode_type", type=str, default="direct", help="select mode type: direct or woodbury")

    args = parser.parse_args()

    for seed in [args.seeds]:
        arg_list = [
            "--mode", args.mode,
            "--eta", str(args.eta),
            "--beta", str(args.beta),
            "--dataset", args.dataset,
            "--n_tasks", str(args.n_tasks),
            "--n", str(args.n),
            "--m", str(args.m),
            "--rnd_seed", str(seed),
            "--model_name", args.model_name,
            "--opt_name", args.opt_name,
            "--sched_name", args.sched_name,
            "--lr", str(args.lr),
            "--batchsize", str(args.batchsize),
            "--memory_size", str(args.memory_size),
            "--online_iter", str(args.online_iter),
            "--data_dir", args.data_dir,
            "--note", args.note,
            "--eval_period", str(args.eval_period),
            "--n_worker", str(args.n_worker),
            "--rnd_NM", 
            "--buffer_size", str(args.buffer_size),
            "--gamma_main", str(args.gamma_main),
        ]

        if args.gpu_transform:
            arg_list.append("--gpu_transform")
        if args.use_amp:
            arg_list.append("--use_amp")

        #print(f"\n[Running seed={seed}] Full Args:\n{' '.join(arg_list)}\n")
        main(arg_list)


if __name__ == "__main__":
    main_runner()

