
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

from collections import defaultdict
import numpy as np

def tarjan(tree):

    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []
    #-------------------------------------------------------------
    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return
    #-------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

def chuliu_edmonds(scores):
    """"""

    np.fill_diagonal(scores, -float('inf')) # prevent self-loops
    scores[0] = -float('inf')
    scores[0,0] = 0
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)
    #print(scores)
    #print(cycles)
    if not cycles:
        return tree
    else:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # locations of cycle; (t) in [0,1]
        cycle = cycles.pop()
        # indices of cycle in original tree; (c) in t
        cycle_locs = np.where(cycle)[0]
        # heads of cycle in original tree; (c) in t
        cycle_subtree = tree[cycle]
        # scores of cycle in original tree; (c) in R
        cycle_scores = scores[cycle, cycle_subtree]
        # total score of cycle; () in R
        cycle_score = cycle_scores.sum()

        # locations of noncycle; (t) in [0,1]
        noncycle = np.logical_not(cycle)
        # indices of noncycle in original tree; (n) in t
        noncycle_locs = np.where(noncycle)[0]
        #print(cycle_locs, noncycle_locs)

        # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
        metanode_head_scores = scores[cycle][:,noncycle] - cycle_scores[:,None] + cycle_score
        # scores of cycle's potential dependents; (n x c) in R
        metanode_dep_scores = scores[noncycle][:,cycle]
        # best noncycle head for each cycle dependent; (n) in c
        metanode_heads = np.argmax(metanode_head_scores, axis=0)
        # best cycle head for each noncycle dependent; (n) in c
        metanode_deps = np.argmax(metanode_dep_scores, axis=1)

        # scores of noncycle graph; (n x n) in R
        subscores = scores[noncycle][:,noncycle]
        
        # pad to contracted graph; (n+1 x n+1) in R
        subscores = np.pad(subscores, ( (0,1) , (0,1) ), 'constant')
        # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
        subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
        # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
        subscores[:-1,-1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]

        # MST with contraction; (n+1) in n+1
        contracted_tree = chuliu_edmonds(subscores)
        # head of the cycle; () in n
        #print(contracted_tree)
        cycle_head = contracted_tree[-1]
        # fixed tree: (n) in n+1
        contracted_tree = contracted_tree[:-1]
        # initialize new tree; (t) in 0
        new_tree = -np.ones_like(tree)
        #print(0, new_tree)
        # fixed tree with no heads coming from the cycle: (n) in [0,1]
        contracted_subtree = contracted_tree < len(contracted_tree)
        # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
        new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
        #print(1, new_tree)
        # fixed tree with heads coming from the cycle: (n) in [0,1]
        contracted_subtree = np.logical_not(contracted_subtree)
        # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
        new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
        #print(2, new_tree)
        # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
        new_tree[cycle_locs] = tree[cycle_locs]
        #print(3, new_tree)
        # root of the cycle; (n)[() in n] in c = () in c
        cycle_root = metanode_heads[cycle_head]
        # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
        new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
        #print(4, new_tree)
        return new_tree

#===============================================================
def chuliu_edmonds_one_root(scores):
    """"""
    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0]+1
    if len(roots_to_try) == 1:
        return tree

    #-------------------------------------------------------------
    def set_root(scores, root):
        root_score = scores[root,0]
        scores = np.array(scores)
        scores[1:,0] = -float('inf')
        scores[root] = -float('inf')
        scores[root,0] = 0
        return scores, root_score
    #-------------------------------------------------------------

    best_score, best_tree = -np.inf, None # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = (tree_probs).sum()+(root_score) if (tree_probs > -np.inf).all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree

def mst(arc_probs, rel_probs=None, use_chi_liu_edmonds=True):
    if use_chi_liu_edmonds:
        arcs = chuliu_edmonds_one_root(arc_probs)
    else:
        arcs = _arc_argmax(arc_probs)
    arcs[0] = 0
    if rel_probs is not None:
        rels = _rel_argmax(rel_probs[np.arange(len(arcs)), arcs])
        rels[0] = -1
    else:
        rels = None
    arcs[0] = -1
    return arcs, rels


def _arc_argmax(probs):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L532  # NOQA
    """
    length = probs.shape[0]
    probs = probs * (1 - np.eye(length))
    heads = np.argmax(probs, axis=1)
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_probs = probs[tokens, 0]
        head_probs = probs[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_probs / head_probs)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_probs = probs[roots, 0]
        probs[roots, 0] = 0
        new_heads = np.argmax(probs[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(probs[roots, new_heads] / root_probs)]
        heads[roots] = new_heads
        heads[new_root] = 0
    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_probs = probs[cycle, old_heads]
        non_heads = np.array(list(dependents))
        probs[np.repeat(cycle, len(non_heads)),
              np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(probs[cycle][:, tokens], axis=1) + 1
        new_probs = probs[cycle, new_heads] / old_probs
        change = np.argmax(new_probs)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)
    return heads


def _rel_argmax(probs, root_rel=1):
    """
    https://github.com/tdozat/Parser-v1/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L612  # NOQA
    """
    length = probs.shape[0]
    tokens = np.arange(1, length)
    rels = np.argmax(probs, axis=1)
    roots = np.where(rels[tokens] == root_rel)[0] + 1
    if len(roots) < 1:
        rels[1 + np.argmax(probs[tokens, root_rel])] = root_rel
    elif len(roots) > 1:
        root_probs = probs[roots, root_rel]
        probs[roots, root_rel] = 0
        new_rels = np.argmax(probs[roots], axis=1)
        new_probs = probs[roots, new_rels] / root_probs
        new_root = roots[np.argmin(new_probs)]
        rels[roots] = new_rels
        rels[new_root] = root_rel
    return rels


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    index = 0
    stack = []
    indices = {}
    lowlinks = {}
    onstack = defaultdict(lambda: False)
    SCCs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in edges[v]:
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif onstack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            SCC = set()
            while True:
                w = stack.pop()
                onstack[w] = False
                SCC.add(w)
                if not(w != v):
                    break
            SCCs.append(SCC)

    for v in vertices:
        if v not in indices:
            strongconnect(v)

    return [SCC for SCC in SCCs if len(SCC) > 1]


import numpy as np
import re

def decode_MST(energies, lengths, leading_symbolic=0, labeled=True):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: numpy 4D tensor
        energies of each edge. the shape is [batch_size, num_labels, n_steps, n_steps],
        where the summy root is at index 0.
    :param masks: numpy 2D tensor
        masks in the shape [batch_size, n_steps].
    :param leading_symbolic: int
        number of symbolic dependency types leading in type alphabets)
    :return:
    """

    #np.save('energies',energies)
    #np.save('lengths',lengths)
    
    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if findcycle:
                break

            if added[i] or not curr_nodes[i]:
                continue

            # init cycle
            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i] = True
            findcycle = True
            l = i

            while par[l] not in tmp_cycle:
                l = par[l]
                if added[l]:
                    findcycle = False
                    break
                added[l] = True
                tmp_cycle.add(l)

            if findcycle:
                lorg = l
                cycle.add(lorg)
                l = par[lorg]
                while l != lorg:
                    cycle.add(l)
                    l = par[l]
                break

        return findcycle, cycle

    def chuLiuEdmonds():
        par = np.zeros([length], dtype=np.int32)
        # create best graph
        par[0] = -1
        for i in range(1, length):
            # only interested at current nodes
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue

                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)
        # no cycles, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue

                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr
            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        for id, cyc_node in enumerate(cycle):
            cyc_nodes[id] = cyc_node
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1

            for j in cyc_nodes:
                if score_matrix[j, i] > max1:
                    max1 = score_matrix[j, i]
                    wh1 = j

                scr = cyc_weight + score_matrix[i, j] - score_matrix[par[j], j]

                if scr > max2:
                    max2 = scr
                    wh2 = j

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)

        for cyc_node in cyc_nodes[1:]:
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != wh:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    if labeled:
        assert energies.ndim == 4, 'dimension of energies is not equal to 4'
    else:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    input_shape = energies.shape
    batch_size = input_shape[0]
    max_length = input_shape[2]

    pars = np.zeros([batch_size, max_length], dtype=np.int32)
    types = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None
    for i in range(batch_size):
        energy = energies[i]

        # calc the realy length of this instance
        length = lengths[i]

        # calc real energy matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic types).
        if labeled:
            energy = energy[leading_symbolic:, :length, :length]
            energy = energy - energy.min() + 1e-6
            # get best label for each edge.
            label_id_matrix = energy.argmax(axis=0) + leading_symbolic
            energy = energy.max(axis=0)
        else:
            energy = energy[:length, :length]
            energy = energy - energy.min() + 1e-6
            label_id_matrix = None
        # get original score matrix
        orig_score_matrix = energy
        # initialize score matrix to original score matrix
        score_matrix = np.array(orig_score_matrix, copy=True)

        oldI = np.zeros([length, length], dtype=np.int32)
        oldO = np.zeros([length, length], dtype=np.int32)
        curr_nodes = np.zeros([length], dtype=np.bool)
        reps = []

        for s in range(length):
            orig_score_matrix[s, s] = 0.0
            score_matrix[s, s] = 0.0
            curr_nodes[s] = True
            reps.append(set())
            reps[s].add(s)
            for t in range(s + 1, length):
                oldI[s, t] = s
                oldO[s, t] = t

                oldI[t, s] = t
                oldO[t, s] = s

        final_edges = dict()
        chuLiuEdmonds()
        par = np.zeros([max_length], np.int32)
        if labeled:
            type = np.ones([max_length], np.int32)
            type[0] = 0
        else:
            type = None

        for ch, pr in final_edges.items():
            par[ch] = pr
            if labeled and ch != 0:
                type[ch] = label_id_matrix[pr, ch]

        par[0] = 0
        pars[i] = par
        if labeled:
            types[i] = type

    #np.save('pars',pars)
    #np.save('types',types)
    #torch.zz
    return pars, types
    