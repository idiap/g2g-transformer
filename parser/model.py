
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


from parser.metric import Metric
import torch
import torch.nn as nn
import numpy as np
import numpy
from tqdm import tqdm
from parser.utils.mst import mst
import torch.nn.functional as F

class Model(object):

    def __init__(self, vocab, parser, config, num_labels):
        super(Model, self).__init__()

        self.vocab = vocab
        self.parser = parser
        self.num_labels = num_labels
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
    
    def prepare_argmax(self, s_arc, s_rel, mask, sbert_arc, sbert_rel, stop_sign, just_pred, mask_gold=None):

        batch_size, max_len = mask.shape

        if mask_gold is not None:
            pred_arcs, pred_rels = s_arc[mask_gold], s_rel[mask_gold]
        else:
            pred_arcs, pred_rels = self.decode(s_arc.clone(), s_rel.clone(), mask.clone())
        
        matrix_arc = torch.zeros((batch_size,max_len)).long().to(mask.device)
        matrix_rel = torch.zeros((batch_size,max_len)).long().to(mask.device)
        
        lengths = mask.sum(dim=1)
        counter = 0
        for i,mask_instance in enumerate(mask):
            lens = len(mask_instance.nonzero())
            if lens != 0:
                matrix_arc[i,mask_instance] = pred_arcs[counter:counter+lens]
                matrix_rel[i,mask_instance] = pred_rels[counter:counter+lens]
                matrix_rel[i,1] = self.vocab.root_label
                counter += lens
                
        assert counter == len(pred_arcs)

        if self.config.subword_option == "syntax":
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_syn(matrix_arc,matrix_rel, mask.clone(), sbert_rel)
        else:
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_sem(matrix_arc, matrix_rel, mask.clone(), sbert_arc, sbert_rel)

        if just_pred:
            return pred_arcs, graph_rel

        if self.config.input_labeled_graph:
            graph_arc,pred_arcs, pred_rels, graph_rel = self.build_graph_labeled(pred_arcs,graph_rel,mask_new, stop_sign)
        else:
            graph_arc,pred_arcs, pred_rels, graph_rel = self.build_graph_unlabeled(pred_arcs,graph_rel,mask_new, stop_sign)

        if mask_gold is None:
            return graph_arc, graph_rel
        else:
            return graph_arc, pred_arcs, pred_rels, graph_rel

    def prepare_mst(self, s_arc, s_rel, mask, sbert_arc, sbert_rel, stop_sign, just_pred, mask_gold=None):

        if mask_gold is not None:
            batch_size, max_len = mask.shape
            p_arcs, pred_rels = s_arc[mask_gold], s_rel[mask_gold]
            pred_arcs = torch.zeros((batch_size, max_len)).long().to(mask.device)
            graph_rel = torch.zeros((batch_size, max_len)).long().to(mask.device)

            counter = 0
            for i, mask_instance in enumerate(mask):
                lens = len(mask_instance.nonzero())
                if lens != 0:
                    pred_arcs[i, mask_instance] = p_arcs[counter:counter + lens]
                    graph_rel[i, mask_instance] = pred_rels[counter:counter + lens]
                    graph_rel[i, 1] = self.vocab.root_label
                    counter += lens
        else:
            pred_arcs, graph_rel = self.decode_mst(s_arc.clone(), s_rel.clone(), mask.clone(), prepare=True)

        if self.config.subword_option == "syntax":
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_syn(pred_arcs,graph_rel, mask.clone(), sbert_rel)
        else:
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_sem(pred_arcs, graph_rel, mask.clone(), sbert_arc, sbert_rel)

        if just_pred:
            return pred_arcs,graph_rel

        if self.config.input_labeled_graph:
            graph_arc,pred_arcs, pred_rels, graph_rel = self.build_graph_labeled(pred_arcs,graph_rel,mask_new, stop_sign)
        else:
            graph_arc, pred_arcs, pred_rels, graph_rel = self.build_graph_unlabeled(pred_arcs,graph_rel,mask_new, stop_sign)

        if mask_gold is None:
            return graph_arc, graph_rel
        else:
            return graph_arc, pred_arcs, pred_rels, graph_rel

    def build_bert_graphs_sem(self, pred_arcs, pred_rels, mask, sbert_arc, sbert_rel):

        sbert_arc[mask] = pred_arcs[mask]
        sbert_arc[:,1] = 0
        mask_new = sbert_rel == self.vocab.sbert_label
        mask_new[:,1] = False

        sbert_rel[mask] = pred_rels[mask]
        sbert_rel[:,1] = self.vocab.root_label

        return sbert_arc, sbert_rel, mask_new

    def build_bert_graphs_syn(self, pred_arcs, pred_rels, mask, sbert_rel):

        mask_new = sbert_rel == self.vocab.sbert_label
        mask_new[:, 1] = False
        mask_total = mask_new.long() + mask.long()

        new_graphs_arc = torch.zeros(mask_total.shape[0],mask_total.shape[1]).long().to(pred_arcs.device)
        new_graphs_rel = torch.zeros(mask_total.shape[0],mask_total.shape[1]).long().to(pred_arcs.device)
        for i,(pred_arc,pred_rel,mask_instance) in enumerate(zip(pred_arcs,pred_rels,mask_total)):
            for j,m in enumerate(mask_instance):
                if m==2:
                    new_graphs_arc[i,j] = pred_arc[j]
                    new_graphs_rel[i,j] = pred_rel[j]
                elif m==1:
                    new_graphs_arc[i,j] = new_graphs_arc[i][j-1]
                    new_graphs_rel[i,j] = new_graphs_rel[i][j-1]
        return new_graphs_arc, new_graphs_rel, mask_new

    def build_graph_unlabeled(self, pred_arcs, graph_rel, mask, stop_sign):
    
        graph_arc = torch.zeros(mask.shape[0],mask.shape[1],mask.shape[1]).long().to(mask.device)
        
        mask = mask.long()
        
        mask = stop_sign.unsqueeze(1) * mask
        
        lengths = mask.sum(dim=1)
        
        graph_rel = graph_rel * mask

        for i,(arc,lens,mask_instance) in enumerate(zip(pred_arcs,lengths,mask)):
            
            if lens != 0:
                graph_arc[i,torch.arange(mask.shape[1]),arc] = 1
                graph_arc[i,:,:] = graph_arc[i,:,:] * mask[i].unsqueeze(1)
                graph_arc[i,:,:] = graph_arc[i,:,:] + 2 * graph_arc[i,:,:].transpose(0,1)

                if not self.config.use_mst_train:
                    graph_arc[i,:,:] = graph_arc[i,:,:] * (graph_arc[i,:,:] != 3)
                assert not len( (graph_arc[i,:,:] == 3).nonzero() )
            if self.config.train_zero_rel:
                mask_instance[1] = 1
                mask_new = (mask_instance.unsqueeze(0) * mask_instance.unsqueeze(1)).bool()
                ## 3 is padding_idx
                graph_arc[i,:,:][~mask_new] = 3

        return graph_arc, pred_arcs, graph_rel, graph_rel
    
    def build_graph_labeled(self, pred_arcs, pred_rel, mask, stop_sign):

        graph_arc = torch.zeros(mask.shape[0],mask.shape[1],mask.shape[1]).long().to(mask.device)
        mask = mask.long()
        
        mask = stop_sign.unsqueeze(1) * mask
        
        lengths = mask.sum(dim=1)
        
        for i,(arc,rel,lens, mask_instance) in enumerate(zip(pred_arcs,pred_rel,lengths,mask)):
            
            if lens != 0:
                graph_arc[i,torch.arange(mask.shape[1]),arc] = rel + 1
                graph_arc[i,:,:] = graph_arc[i,:,:] * mask[i].unsqueeze(1)
            
                assert not( len( (graph_arc[i,:,:] < 0).nonzero() ) or len( (graph_arc[i,:,:] > 
                                                                                   self.num_labels).nonzero() ) ) 
                mask_t = (graph_arc[i,:,:] > 0)*1
                graph_arc_t = graph_arc[i,:,:].transpose(0,1) + mask_t.transpose(0,1) * self.num_labels
                graph_arc[i,:,:] = graph_arc[i,:,:] + graph_arc_t
            
                if not self.config.use_mst_train:
                    mask_t = mask_t + mask_t.transpose(0,1)
                    mask_t = mask_t * (mask_t != 2).long() 
                    graph_arc[i,:,:] = graph_arc[i,:,:] * mask_t

                assert not len( (graph_arc[i,:,:] > 2*self.num_labels).nonzero() )

            if self.config.train_zero_rel:
                mask_instance[1] = 1
                mask_new = (mask_instance.unsqueeze(0) * mask_instance.unsqueeze(1)).bool()
                graph_arc[i,:,:][~mask_new] = 2 * self.num_labels + 1
                

        return graph_arc, pred_arcs, pred_rel, None    
    
    def check_stop(self, stop_sign, arc_new, arc_prev, rel_new, rel_prev, mask):
        
        for i,(mask_instance,narc,parc,nrel,prel) in enumerate(zip(mask,arc_new,arc_prev,rel_new,rel_prev)):
            if len(mask_instance.nonzero()) != 0:
                dif_arc = len( (narc[mask_instance]-parc[mask_instance]).nonzero() ) 
                dif_rel = len( (nrel[mask_instance]-prel[mask_instance]).nonzero() )
                if self.config.stop_arc and dif_arc==0:
                    stop_sign[i] = 0
                elif self.config.stop_arc_rel and dif_arc==0 and dif_rel==0:
                    stop_sign[i] = 0
        
        return stop_sign

    def train_predicted(self, loader):
        self.parser.train()
        pbar = tqdm(total=len(loader))

        for words, tags, arcs, rels, initial_heads, initial_rels, mask, sbert_arc, sbert_rel in loader:

            mask_gold = arcs > 0
            stop_sign = torch.ones(len(words)).long().to(words.device)

            ## iterate over encoder
            for counter in range(1,self.config.num_iter_encoder):
                if counter == 1:
                    graph_arc, prev_arcs, prev_rels, graph_rel = \
                            self.prepare_mst(initial_heads, initial_rels, mask, sbert_arc.clone(),
                                             sbert_rel.clone(), stop_sign,False,mask_gold)
                else:
                    if self.config.use_mst_train:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)

                s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)
                s_arc_t = s_arc[mask]
                s_rel_t = s_rel[mask, :]
                gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

                if self.config.use_two_opts:
                    self.optimizer_nonbert.zero_grad()
                    self.optimizer_bert.zero_grad()
                else:
                    self.optimizer.zero_grad()

                loss = self.get_loss(s_arc_t, s_rel_t, gold_arcs, gold_rels)
                loss.backward()

                if self.config.use_two_opts:
                    self.optimizer_nonbert.step()
                    self.optimizer_bert.step()
                    self.scheduler_nonbert.step()
                    self.scheduler_bert.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()


                if self.config.use_mst_train or counter==1:
                    new_arcs,new_rels = self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),
                                                               sbert_rel.clone(),stop_sign,True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),
                                                                         sbert_rel.clone(),stop_sign,True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        #print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()
                    mask_gold = (stop_sign.unsqueeze(1) * mask_gold.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels


            pbar.update(1)


    def train(self, loader):
        self.parser.train()
        pbar = tqdm(total= len(loader))

        for words, tags, arcs, rels, mask, sbert_arc, sbert_rel in loader:
            
            mask_gold = arcs > 0
            stop_sign = torch.ones(len(words)).long().to(words.device)         
            
            ## iterate over encoder
            for counter in range(self.config.num_iter_encoder):
                if counter==0:
                    s_arc, s_rel = self.parser(words, tags)
                else:
                    if self.config.use_mst_train or counter==1:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    s_arc,s_rel = self.parser(words,tags,stop_sign,graph_arc,graph_rel)
                s_arc_t = s_arc[mask]
                s_rel_t = s_rel[mask,:]
                gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

                if self.config.use_two_opts:
                    self.optimizer_nonbert.zero_grad()
                    self.optimizer_bert.zero_grad()
                else:
                    self.optimizer.zero_grad()
                    
                loss = self.get_loss(s_arc_t, s_rel_t, gold_arcs, gold_rels)
                loss.backward()

                if self.config.use_two_opts:
                    self.optimizer_nonbert.step()
                    self.optimizer_bert.step()
                    self.scheduler_nonbert.step()
                    self.scheduler_bert.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()


                if self.config.use_mst_train or counter==1:
                    new_arcs,new_rels = self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),
                                                               sbert_rel.clone(),stop_sign,True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),
                                                                         sbert_rel.clone(),stop_sign,True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        #print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()
                    mask_gold = (stop_sign.unsqueeze(1) * mask_gold.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

            pbar.update(1)

    @torch.no_grad()
    def evaluate_predicted(self, loader, punct=False):
        self.parser.eval()

        loss, metric = 0, Metric()
        pbar = tqdm(total=len(loader))
        for words, tags, arcs, rels, initial_heads, initial_rels, mask, sbert_arc, sbert_rel in loader:

            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs > 0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(1,self.config.num_iter_encoder):

                self.counter_ref = counter

                if counter == 1:
                    graph_arc, prev_arcs, prev_rels, graph_rel = \
                            self.prepare_mst(initial_heads, initial_rels, mask, sbert_arc.clone(),
                                             sbert_rel.clone(), stop_sign,False,mask_gold)
                else:
                    if self.config.use_mst_train:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)


                s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)

                if self.config.use_mst_train or counter == 1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 1:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                if counter == 1:
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    index = stop_sign.nonzero().squeeze(1)
                    s_arc_final[index] = s_arc[index]
                    s_rel_final[index] = s_rel[index]

                if self.config.show_refinement:
                    if counter == 0:
                        self.initial_refinement_total(new_arcs.clone(), new_rels.clone(), mask_unused.clone()
                                                      , arcs.clone(), rels.clone(), mask_gold.clone())
                    elif counter > 0:
                        self.refinement_total(new_arcs.clone(), new_rels.clone(), mask_unused.clone(),
                                              stop_sign.clone())

            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]
            if self.config.use_mst_eval:
                pred_arcs, pred_rels = self.decode_mst(s_arc_final,s_rel_final, mask_unused,prepare=False)
            else:
                pred_arcs, pred_rels = self.decode(s_arc_final, s_rel_final, mask_unused)

            s_arc_mask = s_arc_final[mask_unused]
            s_rel_mask = s_rel_final[mask_unused]

            loss += self.get_loss(s_arc_mask, s_rel_mask, gold_arcs, gold_rels)

            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
            pbar.update(1)

        loss /= len(loader)
        return loss, metric


    @torch.no_grad()
    def evaluate(self, loader, punct=False):
        self.parser.eval()

        loss, metric = 0, Metric()
        pbar = tqdm(total=len(loader))
        for words, tags, arcs, rels, mask, sbert_arc, sbert_rel in loader:
            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs>0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(self.config.num_iter_encoder):

                self.counter_ref = counter

                if counter==0:
                    s_arc, s_rel = self.parser(words, tags)
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    if self.config.use_mst_train or counter==1:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)

                    s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)


                if self.config.use_mst_train or counter==1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                index = stop_sign.nonzero()
                s_arc_final[index] = s_arc[index]
                s_rel_final[index] = s_rel[index]

                if self.config.show_refinement:
                    if counter == 0:
                        self.initial_refinement_total(new_arcs.clone(), new_rels.clone(), mask_unused.clone()
                                          , arcs.clone(), rels.clone(), mask_gold.clone())
                    elif counter > 0:
                        self.refinement_total(new_arcs.clone(), new_rels.clone(), mask_unused.clone(),
                                              stop_sign.clone())

            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

            if self.config.use_mst_eval:
                pred_arcs, pred_rels = self.decode_mst(s_arc_final,s_rel_final, mask_unused,prepare=False)
            else:
                pred_arcs, pred_rels = self.decode(s_arc_final, s_rel_final,mask_unused)


            s_arc_mask = s_arc_final[mask_unused]
            s_rel_mask = s_rel_final[mask_unused]
            
            loss += self.get_loss(s_arc_mask, s_rel_mask, gold_arcs, gold_rels)
            
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
            pbar.update(1)

        loss /= len(loader)
        return loss, metric

    @torch.no_grad()
    def predict_predicted(self, loader):
        self.parser.eval()
        metric = Metric()
        all_arcs, all_rels = [], []


        for words, tags, arcs, rels, initial_heads, initial_rels, mask, sbert_arc, sbert_rel in loader:

            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs > 0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(1,self.config.num_iter_encoder):
                if counter == 1:
                    graph_arc, prev_arcs, prev_rels, graph_rel = \
                            self.prepare_mst(initial_heads, initial_rels, mask, sbert_arc.clone(),
                                             sbert_rel.clone(), stop_sign,False,mask_gold)
                else:
                    if self.config.use_mst_train:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)


                s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)

                if self.config.use_mst_train or counter==1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 1:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                if counter == 1:
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    index = stop_sign.nonzero().squeeze(1)
                    s_arc_final[index] = s_arc[index]
                    s_rel_final[index] = s_rel[index]



            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

            if self.config.use_mst_eval:
                pred_rels, pred_arcs_org, pred_arcs = self.decode_mst(s_arc_final, s_rel_final,
                                                           mask_unused,prepare=False,do_predict=True)


            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

            lens = mask_unused.sum(1).tolist()

            all_arcs.extend(torch.split(pred_arcs_org, lens))
            all_rels.extend(torch.split(pred_rels, lens))

        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels, metric


    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        metric = Metric()
        all_arcs, all_rels = [], []
        for words, tags, arcs, rels, mask, sbert_arc, sbert_rel, offsets in loader:

            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs > 0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(0,self.config.num_iter_encoder):

                self.counter_ref = counter

                if counter == 0:
                    s_arc, s_rel = self.parser(words, tags)
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    if self.config.use_mst_train or counter==1:
                        graph_arc, graph_rel = \
                            self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(), sbert_rel.clone(), stop_sign, False)
                    else:
                        graph_arc, graph_rel = \
                            self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(), sbert_rel.clone(), stop_sign,
                                                False)
                    s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)

                if self.config.use_mst_train or counter==1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                index = stop_sign.nonzero()
                s_arc_final[index] = s_arc[index]
                s_rel_final[index] = s_rel[index]

            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

            if self.config.use_mst_eval:
                pred_rels, pred_arcs_org,pred_arcs = self.decode_mst(s_arc_final, s_rel_final,
                                                           mask_unused,prepare=False,do_predict=True)


            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

            lens = mask_unused.sum(1).tolist()

            all_arcs.extend(torch.split(pred_arcs_org, lens))
            all_rels.extend(torch.split(pred_rels, lens))

        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels, metric

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss
  
    def decode(self, s_arc, s_rel, mask):
        
        mask_new = self._mask_arc(s_arc,mask.clone())
        
        s_arc = s_arc + (1. - mask_new) * (-1e8)
        
        s_arc = F.softmax(s_arc,dim=2)        
         
        s_arc = s_arc * mask_new
        
        
        s_arc = s_arc[mask]
        s_rel = s_rel[mask]

        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels

    def re_map(self,pred_arc,mask,do_predict=False):

        dict = {-1:0}
        indicies = mask.nonzero()
        for counter, index in enumerate(indicies):
            dict[counter] = int(index)
        pred_arc_org = np.zeros(len(mask))

        if not do_predict:
            for i in range(len(pred_arc)):
                pred_arc[i] = dict[pred_arc[i]]

        pred_arc_org[mask.bool().cpu().numpy()] = pred_arc

        pred_arc_org *= mask.bool().cpu().numpy()

        return pred_arc_org
        
    def _mask_arc(self, logits_arc, all_mask):
        mask_new = torch.zeros(logits_arc.shape).to(logits_arc.device)
        self_loop = (1-torch.eye(logits_arc.shape[2])).to(logits_arc.device)
        all_mask = all_mask.long()
        for i, mask in enumerate(all_mask):
            mask_new[i,:,:] = mask.unsqueeze(0) * mask.unsqueeze(1)
            mask_new[i,:,1] = 1
            mask_new[i,:,:] *= self_loop
                
        return mask_new.to(logits_arc.device)  

    def decode_mst(self, s_arc, s_rel, mask, prepare, do_predict=False):
        
        mask_new = self._mask_arc(s_arc,mask.clone())
        
        s_arc = s_arc + (1. - mask_new) * (-1e8)

        s_arc = F.softmax(s_arc,dim=2)
        
        s_arc = s_arc*mask_new
        
        
        s_arc_final = np.zeros((mask.shape[0],mask.shape[1]))
        if do_predict:
            s_arc_final_org = np.zeros((mask.shape[0],mask.shape[1]))
        s_rel_final = torch.zeros((mask.shape[0],mask.shape[1])).long().to(mask.device)
        
        for counter, (s_arc_batch,s_rel_batch,mask_instance) in enumerate(zip(s_arc,s_rel,mask)):
            
            if len(mask_instance.nonzero())!= 0:
                
                ## set root mask True
                mask_instance[1] = True
                ## predict the dependencies on word level
                s_arc_batch = s_arc_batch[mask_instance.unsqueeze(0) * mask_instance.unsqueeze(1)].reshape(mask_instance.sum(),mask_instance.sum())
                
                if prepare:
                    pred_arc,_ = mst(s_arc_batch.cpu().detach().numpy(), use_chi_liu_edmonds=True)
                else:
                    pred_arc,_ = mst(s_arc_batch.cpu().numpy(), use_chi_liu_edmonds=True)
               
                
                ## covnert back to original positions
                if do_predict:
                    pred_arc_org = self.re_map(pred_arc, mask_instance.clone(),True)

                pred_arc = self.re_map(pred_arc,mask_instance.clone())


                s_arc_final[counter] = pred_arc

                if do_predict:
                    s_arc_final_org[counter] = pred_arc_org

                s_rel_batch[:,:,self.vocab.root_label] = -1e8
                
                ## predict labels
                pred_rels = s_rel_batch[torch.arange(len(pred_arc)),pred_arc].argmax(dim=-1)
            
                s_rel_final[counter] = pred_rels
                s_rel_final[counter,1] = self.vocab.root_label
                #set root mask False
                mask_instance[1] = False                

            
            
        s_arc_final = torch.from_numpy(s_arc_final).long().to(mask.device)
        if do_predict:
            s_arc_final_org = torch.from_numpy(s_arc_final_org).long().to(mask.device)

        if prepare:
            return s_arc_final, s_rel_final
        else:
            if do_predict:
                return s_rel_final[mask], s_arc_final_org[mask], s_arc_final[mask]
            else:
                return s_arc_final[mask], s_rel_final[mask]
        
        
        
      
           
        
        