
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
# the source code is from https://github.com/yzhangcs/parser

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # initialize k centroids randomly
    c, old = x[torch.randperm(len(x))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)

    while old is None or not c.equal(old):
        # handle the empty clusters
        for i in range(k):
            # choose the farthest datapoint from the biggest cluster
            # and move that the empty cluster
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        # update the centroids
        c, old = torch.tensor([x[y.eq(i)].mean() for i in range(k)]), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)
    clusters = [y.eq(i) for i in range(k)]
    clusters = [i.nonzero().view(-1).tolist() for i in clusters if i.any()]
    centroids = [round(x[i].mean().item()) for i in clusters]

    return centroids, clusters


def collate_fn(data):
    reprs = (pad_sequence(i, True) for i in zip(*data))
    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs


class TextSampler(Sampler):

    def __init__(self, lengths, batch_size, n_buckets, shuffle=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        # NOTE: the final bucket count is less than or equal to n_buckets
        self.sizes, self.buckets = kmeans(x=lengths, k=n_buckets)
        self.chunks = [max(size * len(bucket) // self.batch_size, 1)
                       for size, bucket in zip(self.sizes, self.buckets)]

    def __iter__(self):
        # if shuffle, shffule both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)):
            for batch in range_fn(len(self.buckets[i])).chunk(self.chunks[i]):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)


class TextDataset(Dataset):

    def __init__(self, items, n_buckets=1):
        super(TextDataset, self).__init__()

        self.items = items

    def __getitem__(self, index):
        return tuple(item[index] for item in self.items)

    def __len__(self):
        return len(self.items[0])

    @property
    def lengths(self):
        return [len(i.nonzero()) for i in self.items[0]]


def batchify(dataset, batch_size, n_buckets=64, shuffle=False):
    batch_sampler = TextSampler(lengths=dataset.lengths,
                                batch_size=batch_size,
                                n_buckets=n_buckets,
                                shuffle=shuffle)
    loader = DataLoader(dataset=dataset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn)

    return loader, batch_sampler.buckets
