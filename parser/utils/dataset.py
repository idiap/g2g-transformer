
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Alireza Mohammadshahi <alireza.mohammadshahi@idiap.ch>,

#This file is part of g2g-transformer.

#g2g-transformer is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 2 as
#published by the Free Software Foundation.

#g2g-transformer is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with g2g-transformer. If not, see <http://www.gnu.org/licenses/>.

from collections import namedtuple
from typing import Dict, Tuple, List, Any, Callable
from overrides import overrides
from parser.utils.utils_conllu import parse_line, DEFAULT_FIELDS, process_multiword_tokens
from conllu import parse_incr
from allennlp.common.file_utils import cached_path
import logging
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

UD_Sentence = namedtuple(typename='UD_Sentence',
                      field_names=['ids', 'words', 'lemmas', 'upos_tags',
                                   'xpos_tags', 'feats', 'heads', 'rels',
                                   'multiword_ids', 'multiword_forms'],
                      defaults=[None]*10)

class UniversalDependenciesDatasetReader(object):
    def __init__(self):
        self.sentences = []
        self.ids = []
        self.ROOT = "<ROOT>"

    def load(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = 1
        with open(file_path, 'r') as conllu_file:

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and we replace these word ids with None in process_multiword_tokens.
                annotation = process_multiword_tokens(annotation)               
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]     
                annotation = [x for x in annotation if x["id"] is not None]

                if len(annotation) == 0:
                    continue

                def get_field(tag: str, map_fn: Callable[[Any], Any] = None) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]

                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]

                words = get_field("form")
                lemmas = get_field("lemma")
                upos_tags = get_field("upostag")
                xpos_tags = get_field("xpostag")
                feats = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                                                     if hasattr(x, "items") else "_")
                heads = get_field("head")
                dep_rels = get_field("deprel")
                sentence = UD_Sentence(ids,words,lemmas,upos_tags,xpos_tags,feats,heads, dep_rels,multiword_ids,multiword_forms)
                self.sentences.append(sentence)
                self.ids.append(counter)
                counter = counter + 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.sentences[item]


    @property
    def words(self):
        return [[self.ROOT] + list(sentence.words) for sentence in self.sentences]

    @property
    def tags(self):
        return [[self.ROOT] + list(sentence.upos_tags) for sentence in self.sentences]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.heads)) for sentence in self.sentences]

    @property
    def rels(self):
        return [[self.ROOT] + list(sentence.rels) for sentence in self.sentences]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(heads=sequence)
                          for sentence, sequence in zip(self.sentences, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(rels=sequence)
                          for sentence, sequence in zip(self.sentences, sequences)]

    def save(self, fname):
        with open(fname, 'w') as f:
            for item in self.sentences:
                output = self.write(item)
                f.write(output)

    def write(self, outputs) :
        outputs = dict(outputs._asdict())
        word_count = len([word for word in outputs["words"]])
        lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
                      for k in ["ids", "words", "lemmas", "upos_tags", "xpos_tags", "feats",
                                "heads", "rels"]])

        multiword_map = None
        if outputs["multiword_ids"]:
            multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
            multiword_forms = outputs["multiword_forms"]
            multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}

        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            # Handle multiword tokens
            if multiword_map and i+1 in multiword_map:
                id_, form = multiword_map[i+1]
                row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
                output_lines.append(row)

            row = "\t".join(line) + "".join(["\t_"] * 2)
            output_lines.append(row)

        return "\n".join(output_lines) + "\n\n"

class UniversalDependenciesRawDatasetReader(UniversalDependenciesDatasetReader):
    def __init__(self, language):
        super().__init__()
        self.tokenizer = SpacyWordSplitter(language=language,pos_tags=True)

    @overrides
    def load(self, file_path):
        file_path = cached_path(file_path)
        counter = 1
        with open(file_path, 'r') as conllu_file:
            for sentence in conllu_file:
                if sentence:
                    words = [word.text for word in self.tokenizer.split_words(sentence)]
                    upos_tags = [word.tag_ for word in self.tokenizer.split_words(sentence)]
                    xpos_tags = upos_tags
                    seq_len = len(words)
                    ids = [i+1 for i in range(seq_len)]
                    lemmas = ["_" for i in range(seq_len)]
                    feats = lemmas
                    heads = [1 for i in range(seq_len)]
                    dep_rels = ["<UNK>" for i in range(seq_len)]
                    multiword_ids = []
                    multiword_forms = []
                    sentence = UD_Sentence(ids,words,lemmas,upos_tags,xpos_tags,feats,heads, dep_rels,multiword_ids,multiword_forms)
                    self.sentences.append(sentence)
                    self.ids.append(counter)
                    counter = counter + 1
