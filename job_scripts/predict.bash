#!/bin/bash

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

# main directory
main_path=""
# initial prediction
initial=""
# original conllu input
input=""
modelname=""
output_path=""
# UD: conllu, penn: conllx, german: other
type="conllu"

if [ ! -d $output_path ]; then
  mkdir -p $output_path;
fi

if [ "$type" = "conllu" ]; then
    echo "Input is CONLL-U format"
    perl $main_path/conllu_to_conllx.pl < $input > $output_path/original.conllx
    perl $main_path/conllu_to_conllx_no_underline.pl < $input > $output_path/original_nounderline.conllx
    if [ "$initial" != "" ]; then
      echo "Initial input is COLL-U format"
      perl $main_path/conllu_to_conllx.pl < $initial > $output_path/original_initial.conllx
      perl $main_path/conllu_to_conllx_no_underline.pl < $initial> $output_path/original_nounderline_initial.conllx
    fi
else
    echo "Input is CONLL-X format"
    cp $input $output_path/original.conllx
    if ["$initial" != "" ]; then
      echo "Initial input is CONLL-X format"
      cp $initial $output_path/original_initial.conllx
    fi
fi

echo "Predicting the input file"
if [ "$initial" = "" ]; then
  python run.py predict --modelname $modelname --fdata $output_path/original.conllx --fpred $output_path/pred.conllx --mainpath $main_path/
else
  python run.py predict --fdata $output_path/original.conllx --finit $output_path/original_initial.conllx --modelname $modelname --use_predicted \
            --fpred $output_path/pred.conllx --mainpath $main_path/
fi
echo "Finished Prediction"

if [ "$type" = "conllu" ]; then
    echo "Converting back to CONLL-U format"
    python substitue_underline.py $output_path/original_nounderline.conllx $output_path/pred.conllx $output_path/pred_nounderline.conllx
    perl $main_path/restore_conllu_lines.pl $output_path/pred_nounderline.conllx $input  > $output_path/pred.conllu
else
    echo "Output is CONLL-X format"
fi

if [ "$type" = "conllu" ]; then
    echo "Evaluating based on official UD script"
    python $main_path/ud_eval.py $input $output_path/pred.conllu -v
else
    if [ "$type" = "conllx" ]; then
        perl eval.pl -g $input -s $output_path/pred.conllx -q
    else
        perl eval.pl -g $input -s $output_path/pred.conllx -q -p
    fi
    echo "done"
fi
