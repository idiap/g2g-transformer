#! /bin/bash
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

MODELNAME="rngtr"
MAIN_PATH=""
TRAIN_PATH=""
DEV_PATH=""
TEST_PATH=""
BERT_PATH=""
# output of initial parser
INIT_TRAIN_PATH=""
INIT_TEST_PATH=""
INIT_DEV_PATH=""
python run.py train --lr1 1e-5 --lr2 2e-3 -w 0.001 --main_path $MAIN_PATH \
                    --modelname $MODELNAME --num_iter_encoder 4 --batch_size 800 --ftrain $TRAIN_PATH --ftest $TEST_PATH --fdev $DEV_PATH \
                    --punct --bert_path $BERT_PATH --input_labeled_graph --use_mst_eval --use_two_opts --layernorm_key \
                    --fpredicted_train $INIT_TRAIN_PATH --fpredicted_dev  $INIT_DEV_PATH --fpredicted_test  $INIT_TEST_PATH\
                    --use_predicted --stop_arc_rel
