
#Copyright (c) 20xx Idiap Research Institute, http://www.idiap.ch/
#Written by Alireza Mohammadshahi <alireza.mohammadshahi@idiap.ch>,

#This file is part of g2g-transformer.

#g2g-transformer is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 3 as
#published by the Free Software Foundation.

#g2g-transformer is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with g2g-transformer. If not, see <http://www.gnu.org/licenses/>.

from collections import Counter
import os
import regex
import torch
from transformers import *

class Vocab(object):
    PAD = '[PAD]'
    UNK = '[UNK]'
    BERT = '[BERT]'

    def __init__(self, config, words, tags, rels):

        self.config = config
        self.batchnorm = config.batchnorm_key or config.batchnorm_value
        self.max_pad_length = config.max_seq_length
        
        self.words = [self.PAD, self.UNK] + sorted(words)
        
        self.tags = sorted(tags)
        self.tags = [self.PAD, self.UNK] + ['<t>:'+tag for tag in self.tags]
        
        self.rels = sorted(rels)
        self.rels = [self.PAD, self.UNK, self.BERT] + self.rels
        self.word_dict = {word: i for i,word in enumerate(self.words)}
        self.punct = [word for word, i in self.word_dict.items() if regex.match(r'\p{P}+$', word)]

        print("Use Normal Bert model")
        self.bertmodel = BertModel.from_pretrained(config.bert_path)

        if config.use_japanese:
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(config.bert_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        self.tokenizer.add_tokens(self.tags + ['<ROOT>']+ self.punct)
        
        self.bertmodel.resize_token_embeddings(len(self.tokenizer))
        # Train our model
        self.bertmodel.train()

        if os.path.exists(config.main_path + "/model" + "/model_" + config.modelname) != True:
            os.mkdir(config.main_path + "/model" + "/model_" + config.modelname)
            
        ### Now let's save our model and tokenizer to a directory
        self.bertmodel.save_pretrained(config.main_path + "/model" + "/model_" + config.modelname)

        self.tag_dict = {tag: i for i,tag in enumerate(self.tags)}
        
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        
        self.root_label = self.rel_dict['<ROOT>']
        self.sbert_label = self.rel_dict['[BERT]']

        self.puncts = []
        for punct in self.punct:
            self.puncts.append(self.tokenizer.convert_tokens_to_ids(punct))
        
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.PAD))[0]
        self.unk_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.UNK))[0]
        
        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words
        self.unk_count = 0
        self.total_count = 0
        self.long_seq = 0

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_tags} tags, "
        info += f"{self.n_rels} rels"

        return info

    def map_arcs_bert_pred(self, corpus, predicted_corpus, training=True):

        all_words = []
        all_tags = []
        all_heads = []
        all_masks = []
        all_rels = []
        all_sbert_arc = []
        all_sbert_rel = []
        all_predicted_head = []
        all_predicted_rel = []

        for i, (words, tags, heads, rels, predicted_heads, predicted_rels) in enumerate(zip(corpus.words,
                                                                            corpus.tags, corpus.heads,corpus.rels,
                                                                            predicted_corpus.heads, predicted_corpus.rels)):

            old_to_new_node = {0: 0}
            tokens_org, tokens_length = self.word2id(words)
            tokens = [item for sublist in tokens_org for item in sublist]

            index = 0
            for token_id, token_length in enumerate(tokens_length):
                index += token_length
                old_to_new_node[token_id + 1] = index

            # CLS heads and tags
            new_heads = []
            new_predicted_heads = []
            new_tags = []
            new_subword_head = [0]

            offsets = torch.tensor(list(old_to_new_node.values()))[:-1] + 1

            for token_id, (offset, token_length) in enumerate(zip(offsets, tokens_length)):
                new_predicted_heads.append(old_to_new_node[predicted_heads[token_id]] + 1)
                new_heads.append(old_to_new_node[heads[token_id]] + 1)
                for sub_token in range(token_length):
                    new_tags.append(tags[token_id])

                    new_subword_head.append(int(offset))
            new_subword_head.append(0)

            new_subword_rel = [self.PAD] + len(tokens) * [self.BERT] + [self.PAD]
            ### add one for CLS #igonore CLS, ROOT, SEP

            words_id = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokens))

            self.unk_count += len((words_id == 100).nonzero())
            self.total_count += len(words_id)
            
            tags = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(self.tag2id(new_tags)))

            masks = torch.zeros(len(words_id)).long()
            masks[offsets[1:]] = 1
            rels = self.rel2id(rels[1:])
            rels_predicted = self.rel2id(predicted_rels[1:])

            if len(masks) < 512:
                all_words.append(words_id)
                all_tags.append(tags)
                all_masks.append(masks.bool())
                assert masks.sum() == len(new_heads[1:])
                all_heads.append(torch.tensor(new_heads[1:]))
                all_predicted_head.append(torch.tensor(new_predicted_heads[1:]))
                all_rels.append(rels)
                all_predicted_rel.append(rels_predicted)
                all_sbert_arc.append(torch.tensor(new_subword_head))
                all_sbert_rel.append(self.rel2id(new_subword_rel))
            else:
                self.long_seq += 1

        print("Percentage of unknown tokens in BERT:{}".format(self.unk_count * 1.0 / self.total_count * 100))
        self.unk_count = 0
        self.total_count = 0

        return all_words, all_tags, all_heads, all_rels, all_predicted_head, all_predicted_rel,\
                all_masks, all_sbert_arc, all_sbert_rel

    
    def map_arcs_bert(self, corpus, training=True):
        
        all_words = []
        all_tags = []
        all_heads = []
        all_masks = []
        all_rels = []
        all_sbert_arc = []
        all_sbert_rel = []

        if not training:
            all_offsets = []

        for i,(words,tags,heads,rels) in enumerate(zip(corpus.words, corpus.tags, corpus.heads, corpus.rels)):

            old_to_new_node = {0:0}
            tokens_org, tokens_length = self.word2id(words)
            tokens = [item for sublist in tokens_org for item in sublist]
            index = 0
            for token_id, token_length in enumerate(tokens_length):
                index += token_length
                old_to_new_node[token_id+1] = index
            # CLS heads and tags
            new_heads = []
            new_tags = []
            new_subword_head = [0]
            
            offsets = torch.tensor(list(old_to_new_node.values() ))[:-1] + 1
            for token_id, (offset, token_length) in enumerate(zip(offsets, tokens_length)):
                new_heads.append(old_to_new_node[heads[token_id]]+1)
                for sub_token in range(token_length):
                    new_tags.append(tags[token_id])
                    
                    new_subword_head.append(int(offset))
            new_subword_head.append(0)
            
            new_subword_rel = [self.PAD] + len(tokens) * [self.BERT] + [self.PAD]

            words_id = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokens))

            self.unk_count += len( (words_id==100).nonzero() )
            self.total_count += len(words_id)

            tags = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(self.tag2id(new_tags) ))
            masks = torch.zeros(len(words_id)).long()
            masks[offsets[1:]] = 1
            rels = self.rel2id(rels[1:])

            if len(masks) < 512:
                all_words.append(words_id)
                all_tags.append(tags)
                all_masks.append(masks.bool())
                assert masks.sum() == len(new_heads[1:])
                all_heads.append(torch.tensor(new_heads[1:]))
                all_rels.append(rels)
                all_sbert_arc.append(torch.tensor(new_subword_head))
                all_sbert_rel.append(self.rel2id(new_subword_rel))
                if not training:
                    assert len(offsets) == len(rels)+1
                    all_offsets.append(offsets[1:])
            else:
                self.long_seq += 1

        print("Percentage of unknown tokens in BERT:{}".format(self.unk_count * 1.0 / self.total_count * 100))
        self.unk_count = 0
        self.total_count = 0

        if not training:
            return all_words,all_tags,all_heads,all_rels,all_masks,all_sbert_arc,all_sbert_rel,all_offsets
        else:
            return all_words,all_tags,all_heads,all_rels,all_masks,all_sbert_arc,all_sbert_rel

    def word2id(self, sequence):
        WORD2ID = []
        lengths = []
        for word in sequence:
            x = self.tokenizer.tokenize(word)
            if len(x) == 0:
                x = ['[UNK]']
            x = self.tokenizer.convert_tokens_to_ids(x)
            lengths.append(len(x))
            WORD2ID.append(x)
        return WORD2ID,lengths   
    
    def tag2id(self, sequence):
        
        tags = []
        for tag in sequence:
            tokenized_tag = self.tokenizer.tokenize('<t>:'+tag)
            if len(tokenized_tag) != 1:
                tags.append(self.unk_index)
            else:
                tags.append(self.tokenizer.convert_tokens_to_ids(tokenized_tag)[0])
        return tags

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 1)
                             for rel in sequence])

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def extend(self, words):
        self.words.extend(sorted(set(words).difference(self.word_dict)))
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)


    def numericalize(self, corpus, predicted_corpus = None, training=True):

        if predicted_corpus is None:
            return self.map_arcs_bert(corpus,training)
        else:
            return self.map_arcs_bert_pred(corpus,predicted_corpus,training)

    @classmethod
    def from_corpus(cls, config, corpus, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        tags = list({tag for seq in corpus.tags for tag in seq})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(config, words, tags, rels)

        return vocab
