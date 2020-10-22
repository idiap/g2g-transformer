
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Alireza Mohammadshahi <alireza.mohammadshahi@idiap.ch>,

# This file is part of g2g-transformer.

# g2g-transformer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.

# g2g-transformer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with g2g-transformer. If not, see <http://www.gnu.org/licenses/>.


import argparse
import os
from parser.cmds import Predict, Train

import torch

from config import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--model', '-m', default='exp/',
                               help='path to model file')
        subparser.add_argument('--vocab', '-v', default='exp/',
                               help='path to vocab file')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=4, type=int,
                               help='max num of threads')
        subparser.add_argument('--batch_size', default=1000, type=int,
                               help='max num of buckets to use')        
        

        
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    print(f"Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(config)
