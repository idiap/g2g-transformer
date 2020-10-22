
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
from os import path
from datetime import datetime, timedelta
from parser import BiaffineParser, Model
from parser.metric import Metric
from parser.utils import Corpus, Vocab
from parser.utils.data import TextDataset, batchify
import torch
from transformers import AdamW,get_linear_schedule_with_warmup

class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--n_mlp_arc', default=500, type=int,
                               help='Dimension of unlabelled dependency classifier')
        subparser.add_argument('--n_mlp_rel', default=100, type=int,
                               help='Dimention of labelled dependency classifier')
        subparser.add_argument('--epochs', default=200, type=int,
                               help='Number of epochs')
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--punct', default=False, action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--ftrain', default='data/train.conll',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/test.conll',
                               help='path to test file')
        subparser.add_argument('--use_predicted', default=False, action='store_true',
                               help='Use predicted parser from outside')
        subparser.add_argument('--fpredicted_train', default='',
                               help='path to predicted parse file (train)')
        subparser.add_argument('--fpredicted_dev', default='',
                               help='path to predicted parse file (dev)')
        subparser.add_argument('--fpredicted_test', default='',
                               help='path to predicted parse file (test)')
        subparser.add_argument('--warmupproportion', '-w', default=0.01, type=float,
                               help='Warm up proportion for BertAdam optimizer')
        subparser.add_argument('--lowercase', default=False, action='store_true',
                               help='whether to do lowercase')    
        subparser.add_argument('--lower_for_nonbert', default=False, action='store_true',
                               help='Divide warmup proportion for non-bert parameters by 2')
        subparser.add_argument('--stop_arc', default=False, action='store_true',
                               help='Stop the refinement steps based on unlabelled graph')
        subparser.add_argument('--stop_arc_rel', default=False, action='store_true',
                               help='Stop the refinement steps based on labelled graph')
        subparser.add_argument('--modelname', default='None',
                               help='path to test file')
        subparser.add_argument('--thr', default=0.001, type=float,
                               help='Threshold for stopping iteration')
        subparser.add_argument('--lr1', default=1e-5, type=float,
                               help='Learning rate for pre-trained parameters (for '
                                    'whole parameters if one optimizer is used)')
        subparser.add_argument('--lr2', default=2e-3, type=float,
                               help='Learning rate for other')
        subparser.add_argument('--num_iter_encoder', default=8, type=int,
                               help='Number of iteration for encoder')
        subparser.add_argument('--same_flag', default=False, action='store_true',
                               help='Use the same baseline and graph models')
        subparser.add_argument('--use_mst_train', default=False, action='store_true',
                               help='Use MST algorithm to decode scores in train model')
        subparser.add_argument('--use_mst_eval', default=False, action='store_true',
                               help='Use MST algorithm to decode scores in eval mode')
        subparser.add_argument('--input_labeled_graph', default=False, action='store_true',
                               help='Input labeled dependency graph')
        subparser.add_argument('--input_unlabeled_graph', default=False, action='store_true',
                               help='Input Unlabeled dependency graph')
        subparser.add_argument("--subword_option", type=str, choices=["syntax", "semantic"],
                            default="semantic")
        subparser.add_argument('--use_japanese', default=False, action='store_true',
                               help='Use Japanese BERT')
        subparser.add_argument('--layernorm_key', default=False, action='store_true',
                               help='layer normalization for Key')  
        subparser.add_argument('--layernorm_value', default=False, action='store_true',
                               help='layer normalization for Value')  
        subparser.add_argument('--use_two_opts', default=False, action='store_true',
                               help='Use one optimizer for Bert and one for others') 
        subparser.add_argument('--mix_layers', default=False, action='store_true',
                               help='Use the mixture of internal layers instead of last layer') 
        subparser.add_argument('--layer_dropout', default=0.1,type=float,
                               help='Layer dropout when mix layers')
        subparser.add_argument('--mlp_dropout', default=0.33,type=float,
                               help='MLP drop out')
        subparser.add_argument('--weight_decay', default=0.01,type=float,
                               help='Weight Decay')
        subparser.add_argument('--max_grad_norm', default=1.0,type=float,
                               help='Clip gradient')
        subparser.add_argument('--bert_path', default='', help='path to BERT')
        subparser.add_argument('--main_path', default='', help='path to main directory')

        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)

        if config.use_predicted:
            train_predicted = Corpus.load(config.fpredicted_train)
            dev_predicted  = Corpus.load(config.fpredicted_dev)
            test_predicted = Corpus.load(config.fpredicted_test)

        if path.exists(config.main_path+"/exp") != True:
            os.mkdir(config.main_path+"/exp")

        if path.exists(config.main_path+"/model") != True:
            os.mkdir(config.main_path+"/model")


        if path.exists(config.main_path+config.model+config.modelname) != True:
            os.mkdir(config.main_path+config.model+config.modelname)
            
        vocab = Vocab.from_corpus(config=config, corpus=train, min_freq=2)
        
        torch.save(vocab, config.vocab+config.modelname + "/vocab.tag")
        
        config.update({
            'n_words': vocab.n_train_words,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        
        print("Load the dataset")


        if config.use_predicted:
            trainset = TextDataset(vocab.numericalize(train,train_predicted))
            devset = TextDataset(vocab.numericalize(dev,dev_predicted))
            testset = TextDataset(vocab.numericalize(test,test_predicted))
        else:
            trainset = TextDataset(vocab.numericalize(train))
            devset = TextDataset(vocab.numericalize(dev))
            testset = TextDataset(vocab.numericalize(test))


        # set the data loaders
        train_loader,_ = batchify(dataset=trainset,
                                batch_size=config.batch_size,
                                n_buckets=config.buckets,
                                shuffle=True)
        dev_loader,_  = batchify(dataset=devset,
                              batch_size=config.batch_size,
                              n_buckets=config.buckets)
        test_loader,_ = batchify(dataset=testset,
                               batch_size=config.batch_size,
                               n_buckets=config.buckets)
        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences in total, "
              f"{len(test_loader):3} batches provided")

        print("Create the model")
        parser = BiaffineParser(config, vocab.n_rels, vocab.bertmodel)

        print("number of pars:{}".format(sum(p.numel() for p in parser.parameters()
                                             if p.requires_grad)))
        if torch.cuda.is_available():
            print('device:cuda')
            device = torch.device('cuda')
            parser = parser.to(device)
        #print(f"{parser}\n")

        model = Model(vocab, parser, config, vocab.n_rels)
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        
        num_train_optimization_steps = int(config.num_iter_encoder * config.epochs * len(train_loader))
        warmup_steps = int(config.warmupproportion * num_train_optimization_steps)

        if config.use_two_opts:
            model_nonbert = []
            model_bert = []
            layernorm_params = ['layernorm_key_layer', 'layernorm_value_layer', 'dp_relation_k', 'dp_relation_v']
            for name, param in parser.named_parameters():
                if 'bert' in name and not any(nd in name for nd in layernorm_params):
                    model_bert.append((name, param))
                else:
                    model_nonbert.append((name, param))

            # Prepare optimizer and schedule (linear warmup and decay) for Non-bert parameters
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters_nonbert = [
                {'params': [p for n, p in model_nonbert if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay},
                {'params': [p for n, p in model_nonbert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            model.optimizer_nonbert = AdamW(optimizer_grouped_parameters_nonbert, lr=config.lr2)

            model.scheduler_nonbert = get_linear_schedule_with_warmup(model.optimizer_nonbert,
                                                                      num_warmup_steps=warmup_steps,
                                                                      num_training_steps=num_train_optimization_steps)

            # Prepare optimizer and schedule (linear warmup and decay) for Bert parameters
            optimizer_grouped_parameters_bert = [
                {'params': [p for n, p in model_bert if not any(nd in n for nd in no_decay)],
                    'weight_decay': config.weight_decay},
                {'params': [p for n, p in model_bert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            model.optimizer_bert = AdamW(optimizer_grouped_parameters_bert, lr=config.lr1)
            model.scheduler_bert = get_linear_schedule_with_warmup(
                model.optimizer_bert, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
            )

        else:
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in parser.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': config.weight_decay},
                {'params': [p for n, p in parser.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            model.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr1)
            model.scheduler = get_linear_schedule_with_warmup(
                model.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
            )

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            if config.use_predicted:
                model.train_predicted(train_loader)
            else:
                model.train(train_loader)
            print(f"Epoch {epoch} / {config.epochs}:")

            if config.use_predicted:
                loss, dev_metric = model.evaluate_predicted(dev_loader, config.punct)
            else:
                loss, dev_metric = model.evaluate(dev_loader, config.punct)

            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            if config.use_predicted:
                loss, test_metric = model.evaluate_predicted(test_loader, config.punct)
            else:
                loss, test_metric = model.evaluate(test_loader, config.punct)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                print(config.model + config.modelname + "/model_weights")
                model.parser.save(config.model + config.modelname + "/model_weights")
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = BiaffineParser.load(config.model + config.modelname + "/model_weights")
        if config.use_predicted:
            loss, metric = model.evaluate_predicted(test_loader, config.punct)
        else:
            loss, metric = model.evaluate(test_loader, config.punct)
        print(metric)
        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
