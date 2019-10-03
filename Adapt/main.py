import os
import sys
import argparse
from lib.config import cfg, cfg_from_file, cfg_from_list
from trainer import Trainer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)

    if cfg.MODEL.SOURCE_ONLY == True:
        trainer.train_src_only()
    else:
        trainer.train()