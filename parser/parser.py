
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


from parser.modules import (MLP, Biaffine, SharedDropout)
import torch
import torch.nn as nn
from parser.utils.graph import BertGraphModel
from parser.utils.scalar_mix import ScalarMixWithDropout
from parser.utils.graph import initialize_bertgraph

class GraphBiaffineParser(nn.Module):

    def __init__(self, base, config, label_size, bertmodel, baseline_parser=None):
        super(GraphBiaffineParser, self).__init__()

        self.config = config

        if base:
            self.bert = initialize_bertgraph(
                config.modelpath + "/model_temp")
        else:
            self.bert = initialize_bertgraph(config.modelpath+"/model_temp",
                                           config.layernorm_key,config.layernorm_value,config.input_labeled_graph,
                                           config.input_unlabeled_graph,label_size)

        if config.mix_layers:
            self.scalar_mix = ScalarMixWithDropout(13,
                                           do_layer_norm=False,
                                           dropout=config.layer_dropout)
        self.mlp_arc_h = MLP(n_in=self.bert.config.hidden_size,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=self.bert.config.hidden_size,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.bert.config.hidden_size,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.bert.config.hidden_size,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        
        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True)
         
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

    def forward(self, words, tags, sign=None,graph_arc=None, graph_rel=None):
        
        mask = words.ne(self.pad_index)
        if sign is not None:
            mask = sign.unsqueeze(1) * mask
            mask = mask.bool()
        x = self.bert(words,mask,tags, graph_arc,graph_rel)

        if self.config.mix_layers:
            x = x[2]
            x = self.scalar_mix(x, mask)
        else:
            x = x[0]

        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get distribution of labelled and unlabelled dependency graph
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        
        return s_arc, s_rel

class BiaffineParser(nn.Module):

    def __init__(self, config, label_size, bertmodel):
        super(BiaffineParser, self).__init__()

        self.config = config
        self.same_flag = config.same_flag
        self.use_label = config.input_labeled_graph
        self.label_size = label_size
        self.bertmodel = bertmodel

        if not(self.same_flag) and not(config.use_predicted):
            self.parser_baseline = GraphBiaffineParser(True,self.config,self.label_size,self.bertmodel)
        
        if config.num_iter_encoder > 1:
            self.parser_graph = GraphBiaffineParser(False,self.config,self.label_size,self.bertmodel)

    def forward(self, words, tags, sign=None, graph_arc=None, graph_rel=None):

        if graph_arc is None:
            if self.same_flag:
                s_arc, s_rel = self.parser_graph(words,tags)
            else:
                s_arc, s_rel = self.parser_baseline(words,tags)
        else:
            s_arc, s_rel = self.parser_graph(words,tags,sign,graph_arc,graph_rel)
                
        return s_arc, s_rel
       

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'],state['label_size'],state['bertmodel'])
        
        parser.load_state_dict(state['state_dict'],strict=False)
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'label_size':self.label_size,
            'bertmodel':self.bertmodel,
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)
