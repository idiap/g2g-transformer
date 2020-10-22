
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

import os
import sys
import logging
import pdb
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')



if len(sys.argv) < 3:
    sys.stderr.write("usage: python substituteConllxPOS.py conllx_file pos_file\n")
    sys.exit(1)
    
class CoNLLReader:
    def __init__(self, file):
        """

        :param file: FileIO object
        """
        self.file = file

    def __iter__(self):
        return self

    def __next__(self):
        sent = self.readsent()
        if sent == []:
            raise StopIteration()
        else:
            return sent

    def readsent(self):
        """
        Assuming CoNLL-U format, where the columns are:
        ID FORM LEMMA UPOSTAG XPOSTAG FEATS HEAD DEPREL DEPS MISC
        """
        sent = []
        row_str = self.file.readline().strip()
        while row_str != "":
            row = {}
            columns = row_str.split("\t")
            row["ID"] = int(columns[0])
            row["FORM"] = columns[1]
            row["LEMMA"] = columns[2] if len(columns) > 2 else "_"
            row["UPOSTAG"] = columns[3] if len(columns) > 3 else "_"
            row["XPOSTAG"] = columns[4] if len(columns) > 4 else "_"
            row["FEATS"] = columns[5] if len(columns) > 5 else "_"
            row["HEAD"] = columns[6] if len(columns) > 6 else "_"
            row["DEPREL"] = columns[7] if len(columns) > 7 else "_"
            row["DEPS"] = columns[8] if len(columns) > 8 else "_"
            row["MISC"] = columns[9] if len(columns) > 9 else "_"
            sent.append(row)
            row_str = self.file.readline().strip()
        return sent

    def close(self):
        self.file.close()
    
    
def write_row(row,f):
    f.write(str(row["ID"]) + "\t")
    f.write(row["FORM"] + "\t")
    f.write(row["LEMMA"] + "\t")
    f.write(row["UPOSTAG"] + "\t")
    f.write(row["XPOSTAG"] + "\t")
    f.write(row["FEATS"] + "\t")
    f.write(str(row["HEAD"]) + "\t")
    f.write(row["DEPREL"] + "\t")
    f.write(row["DEPS"] + "\t")
    f.write(row["MISC"] + "\n")


conll_reader_org = CoNLLReader(open(sys.argv[1]))
conll_reader_pred = CoNLLReader(open(sys.argv[2]))
f = open(sys.argv[3],"w") 

for conll_sent_org, conll_sent_pred in zip(conll_reader_org, conll_reader_pred):
    assert len(conll_sent_org) == len(conll_sent_pred)
    for conll_row_org, conll_row_pred in zip(conll_sent_org, conll_sent_pred):
        conll_row_pred["FORM"] = conll_row_org["FORM"]
        conll_row_pred["LEMMA"] = conll_row_org["LEMMA"]
        write_row(conll_row_pred,f)
    f.write("\n")
    
conll_reader_org.close()
conll_reader_pred.close()
