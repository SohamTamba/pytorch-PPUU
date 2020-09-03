from eval_policy import _main, get_optimal_pool_size
from itertools import chain
import os
import torch
import pickle
import numpy
import argparse

class ArgsHolder:
    def __init__(self):
        self.map = 'i80'
        self.v = '3'
        self.seed = 333333
        self.method = 'policy-MPUR'
        self.batch_size = 1
        self.n_batches = 200
        self.lrt = 0.01
        self.ncond = 20
        self.npred = 30
        self.nexec = 1
        self.n_rollouts = 10
        self.rollout_length = 1
        self.bprob_niter = 1
        self.bprob_lrt = 0.1
        self.bprob_buffer = 1
        self.bprob_save_opt_stats = 1
        self.n_dropout_models = 10
        self.opt_z = 0
        self.opt_a = 1
        self.u_reg = 0.0
        self.u_hinge = 1.0
        self.lambda_l = 0.0
        self.lambda_o = 0.0
        self.graph_density = 0.001
        self.display = 0
        self.debug = False
        self.circular_track = True
        self.safety_factor = 0.0
        self.model_dir = 'models/'

        M1 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
             'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
        M2 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
             'beta=1e-06-zdropout=0.0-gclip=5.0-warmstart=1-seed=1.step200000.model'
        M3 = 'model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-' + \
             'zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.0-gclip=5.0-warmstart=1-seed=1.step200000.model'
        M4 = 'model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-' + \
             'zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
        M5 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
             'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step400000.model'
        self.mfile = M5

        self.value_model = ''
        self.policy_model = ''
        self.save_sim_video = False
        self.enable_tensorboard = False
        self.tensorboard_dir = ''
        self.num_processes = 3
        self.save_grad_vid = False

        self.save_dir = "None"
        self.height = 117
        self.width = 24
        self.h_height = 14
        self.h_width = 3
        self.opt_z = (self.opt_z == 1)
        self.opt_a = (self.opt_a == 1)
        self.no_write = True

        if self.num_processes == -1:
            self.num_processes = get_optimal_pool_size()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--one_exec', action='store_true')
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    opt = ArgsHolder()
    if args.local:
        opt.model_dir = '/load_models/'
        opt.num_processes = 1
    else:
        opt.model_dir = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v14'
    unformatted_policy_net = \
        "MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2" +\
        "-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed={seed}-novaluestep{step}.model"

    out_dir = "dist-stats"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    break_after_one = args.local or args.one_exec

    for step in range(10000, 115000 + 1, 5000):
        for safety_factor in [0.0, 0.33, 0.67, 1.0]:
            opt.safety_factor = safety_factor
            for seed in [1, 2, 3]:
                out_file = f"safety_factor={opt.safety_factor}-seed={seed}-step={step}.p"
                out_path = os.path.join(out_dir, out_file)
                if os.path.isfile(out_path):
                    print(f"{out_path} already present.\nSkipping it", flush=True)
                    continue

                opt.policy_model = unformatted_policy_net.format(seed=seed, step=step)

                print(f"Starting {out_path}", flush=True)
                distance_travelled = _main(opt)

                with open(out_path, 'wb') as f:
                    pickle.dump(distance_travelled, f)

                print(f"Saved file: {out_path}")
                if break_after_one:
                    break
            if break_after_one:
                break
        if break_after_one:
            break





