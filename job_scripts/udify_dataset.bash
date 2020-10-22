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

main_path=""
udify=""
input_data=""


str_array1=("UD_Arabic-PADT" "UD_Basque-BDT" "UD_Chinese-GSD" "UD_English-EWT" "UD_Finnish-TDT" "UD_Hebrew-HTB" "UD_Hindi-HDTB" "UD_Italian-ISDT" "UD_Japanese-GSD" "UD_Korean-GSD" "UD_Russian-SynTagRus" "UD_Swedish-Talbanken" "UD_Turkish-IMST")
str_array2=("ar_padt" "eu_bdt" "zh_gsd" "en_ewt" "fi_tdt" "he_htb" "hi_hdtb" "it_isdt" "ja_gsd" "ko_gsd" "ru_syntagrus" "sv_talbanken" "tr_imst")


for ((i=0;i<${#str_array1[@]};++i)); do
        echo ${str_array1[i]}
        echo ${str_array2[i]}
	if [ ! -d $udify/logs/train/${str_array2[i]} ]; then
		mkdir -p $udify/logs/train/${str_array2[i]};
	fi

        for split in train; do
		echo $split
		python $udify/predict.py $udify/logs/udify-model.tar.gz  $input_data/${str_array1[i]}/${str_array2[i]}-ud-$split.conllu $udify/logs/train/${str_array2[i]}/pred.conllu.udify --eval_file $udify/logs/train/${str_array2[i]}/pred.json
		python $main_path/ud_eval.py $input_data/${str_array1[i]}/${str_array2[i]}-ud-$split.conllu $udify/logs/train/${str_array2[i]}/pred.conllu.udify
		perl $main_path/conllu_to_conllx.pl < $udify/logs/train/${str_array2[i]}/pred.conllu.udify > $udify/logs/train/${str_array2[i]}/pred.conllx.udify
		perl $main_path/conllu_to_conllx.pl < $input_data/${str_array1[i]}/${str_array2[i]}-ud-$split.conllu > $udify/logs/train/${str_array2[i]}/org.conllx.$split
		python $main_path/substitue_underline.py $udify/logs/train/${str_array2[i]}/org.conllx.$split $udify/logs/train/${str_array2[i]}/pred.conllx.udify $udify/logs/train/${str_array2[i]}/pred.conllx.correct.udify.$split
        done
done
