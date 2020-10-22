
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

import torch
from parser import BiaffineParser, Model
from parser.utils import Corpus
from parser.utils.data import TextDataset, batchify



class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--fdata', default='data/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--finit', default='data/test.conllx',
                               help='path to pretrained parser')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')
        subparser.add_argument('--modelname', default='None',
                               help='path to test file')
        subparser.add_argument('--mainpath', default='None',
                               help='path to test file')
        subparser.add_argument('--use_predicted', default=False,action='store_true',
                               help='Use predicted Parser')

        return subparser


    def rearange(self, items, ids):

        indicies = []
        for id in ids:
            for i in id:
                indicies.append(i)
        indicies = sorted(range(len(indicies)), key=lambda k: indicies[k])
        items = [items[i] for i in indicies]
        return items

    def __call__(self, args):
        print("Load the model")

        modelpath = args.mainpath + args.model + args.modelname + "/model_weights"
        vocabpath = args.mainpath + args.vocab + args.modelname + "/vocab.tag"

        config = torch.load(modelpath)['config']


        vocab = torch.load(vocabpath)
        parser = BiaffineParser.load(modelpath)
        model = Model(vocab, parser, config, vocab.n_rels)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata)
        if args.use_predicted:
            corpus_predicted = Corpus.load(args.finit)

        if args.use_predicted:
            dataset = TextDataset(vocab.numericalize(corpus, corpus_predicted))
        else:
            dataset = TextDataset(vocab.numericalize(corpus, training=False))
        # set the data loader
        loader, ids = batchify(dataset, config.batch_size, config.buckets)

        print("Make predictions on the dataset")
        if args.use_predicted:
            heads_pred, rels_pred, metric = model.predict_predicted(loader)
        else:
            heads_pred, rels_pred, metric = model.predict(loader)

        print(metric)
        print(f"Save the predicted result to {args.fpred}")

        heads_pred = self.rearange(heads_pred, ids)
        rels_pred = self.rearange(rels_pred,ids)


        corpus.heads = heads_pred
        corpus.rels = rels_pred
        corpus.save(args.fpred)
