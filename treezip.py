#!/usr/bin/python

# Copyright (c) Andrew Li 2018. This file is licensed under the GPLv3.
# See https://github.com/andrew0x4c/treezip for more information,
# including the full LICENSE file.

import struct
import sys
import argparse
import numpy as np

# QUICK REFERENCE
# Header
# - 8 bytes "TREEZIP\x07" (\x07 so printing a compressed file always beeps)
# - 8 bytes length (wow, 16 EiB? so optimistic)
# - 2 byte version
# - 1 byte feat_addr
# - 1 byte feat_prev
# - 12 bytes unused (align to 16 bytes so it looks nice in xxd)
# Compressed data
# - enough bits to initialize
# - actual tree
# - 0[bit]: that bit
# - 1[feat]: which feature (LSB first)
#   - the number of bits used depends on the number of features left
#     at this level of the tree

# some bitwise util functions

def int_log2(x):
    curr = 0
    while 1 << curr < x: curr += 1
    return curr

def bits(arr, size):
    # can't find a function which does this in numpy
    return (arr[:,np.newaxis] >> np.arange(size)) & 1

def unbits(arr):
    return (arr * (1 << np.arange(arr.shape[1]))).sum(axis=1)

# generating the samples
def get_samples(arr, feat_addr, feat_prev):
    data = bits(arr, 8).flatten()
    init = np.zeros(feat_prev)
    init[:min(data.size, feat_prev)] = data[:feat_prev]
    # copy the feat_prev to initialize, unless there isn't enough,
    # in which case copy all of the data and leave the rest 0
    # (the only case this will happen is if there are no samples)
    addr_bits = bits(np.arange(feat_prev, data.size), feat_addr)
    prev_bits = data[np.arange(feat_prev)[::-1]
        + np.arange(data.size - feat_prev)[:,np.newaxis]]
    feats = np.concatenate([addr_bits, prev_bits], axis=1)
    labels = data[np.arange(feat_prev, data.size)]
    return init, feats, labels

# various util functions for feature selection

def nplogp(p):
    return 0.0 if p == 0 else -p * np.log2(p)

def entropy(data):
    assert data.size
    frac_1 = data.sum() / float(data.size)
    return nplogp(frac_1) + nplogp(1 - frac_1)

def is_pure(data):
    return np.all(data == data[0])

def indices_of_feat(feat, feats):
    where_0 = np.where(feats[:,feat] == 0)[0]
    where_1 = np.where(feats[:,feat] == 1)[0]
    return where_0, where_1

class AmbiguousFeaturesException(Exception):
    # this is used when no feature (out of the ones we look at)
    # can split the population anymore; for example:
    #     0, 1, 0 -> 1
    #     0, 1, 0 -> 0
    # this happens when there aren't detailed enough features to distinguish
    # all the samples (ex. artificially setting -a to be too low)
    pass

def choose_feature(feats, labels, out_of=None):
    if out_of is None: out_of = range(feats.shape[1])
    if is_pure(labels): return None
    num_feats = feats.shape[1]
    scores = {}
    for feat in out_of:
        where_0, where_1 = indices_of_feat(feat, feats)
        if where_0.size == 0 or where_1.size == 0: continue
        # don't consider "useless" splits
        tot_entropy = entropy(labels[where_0]) + entropy(labels[where_1])
        scores[feat] = (tot_entropy, max(where_0.size, where_1.size), feat)
        # minimize entropy, break ties by minimizing the largest split
    if len(scores) == 0:
        # all sets of features are the same (so no split will help),
        # and the set of labels is not pure
        raise AmbiguousFeaturesException()
    min_feat = None
    min_score = (np.inf, np.inf, np.inf)
    # storing all the scores, instead of just the best,
    # makes it easier to debug
    #print scores
    for feat, score in scores.items():
        if score < min_score:
            min_feat = feat
            min_score = score
    return min_feat

# representing a decision tree

class Branch:
    def __init__(self, feature, if_0, if_1):
        self.feature = feature
        self.if_0 = if_0
        self.if_1 = if_1
    def pprint(self, out, depth=0):
        out.write("| " * depth + "#" + str(self.feature) + "\n")
        self.if_0.pprint(out, depth=depth+1)
        self.if_1.pprint(out, depth=depth+1)
    def leaf_count(self):
        return self.if_0.leaf_count() + self.if_1.leaf_count()
    def to_bitstring(self, poss):
        yield 1
        idx = poss.index(self.feature)
        num_bits = int_log2(len(poss))
        curr = idx
        for i in range(num_bits): yield (curr >> i) & 1
        if isinstance(self.if_0, Leaf) and isinstance(self.if_1, Leaf):
            yield 0
            yield self.if_0.val
            yield 0
            # because we will never have a branch with both if_0 and if_1
            # the same, we know that the if_1 case will always be
            # the opposite of the if_0 case; thus, we can just not store
            # the if_1 case.
        else:
            without = poss[:idx] + poss[idx+1:]
            for b in self.if_0.to_bitstring(without): yield b
            for b in self.if_1.to_bitstring(without): yield b
            # if we have already used a feature higher in the tree,
            # we know it is useless to use it again, so we don't allow
            # for those cases to save bits
            # (this often makes pretty significant improvements;
            # try replacing the above line with "without = poss" to see)

class Leaf:
    def __init__(self, val):
        self.val = val
    def pprint(self, out, depth=0):
        out.write("| " * depth + "=" + str(self.val) + "\n")
    def leaf_count(self):
        return 1
    def to_bitstring(self, poss):
        yield 0
        yield self.val

def run_tree_on(tree, feats):
    # we do this nonrecursively to make it O(1) space, and
    # potentially reduce some of the method call overhead
    num_nodes = 1
    # this is used for visualizing the "exceptionalness" of bytes
    while isinstance(tree, Branch):
        if feats[tree.feature]:
            tree = tree.if_1
        else:
            tree = tree.if_0
        num_nodes += 1
    # tree now refers to a leaf
    return tree.val, num_nodes

# actually building the decision tree
def make_tree(feats, labels, out_of=None):
    if out_of is None: out_of = set(range(feats.shape[1]))
    feat = choose_feature(feats, labels, out_of)
    if feat is None:
        assert labels.size > 0
        return Leaf(labels[0])
    else:
        where_0, where_1 = indices_of_feat(feat, feats)
        without = out_of - {feat}
        tree_0 = make_tree(feats[where_0], labels[where_0], without)
        tree_1 = make_tree(feats[where_1], labels[where_1], without)
        return Branch(feat, tree_0, tree_1)

# loading the tree from an iterator (the inverse of tree.to_bitstring())
def tree_from_bitstring(bit_iter, poss, leaf_info=None):
    get = bit_iter.next
    node_type = get()
    if node_type == 0: # leaf
        if leaf_info is None:
            return Leaf(get())
        else:
            return Leaf(leaf_info ^ 1)
    else: # branch
        num_bits = int_log2(len(poss))
        curr = 0
        for i in range(num_bits): curr |= get() << i
        feat = poss[curr]
        without = poss[:curr] + poss[curr+1:]
        if_0 = tree_from_bitstring(bit_iter, without)
        # leaf_info is to deal with the case where a branch has
        # two leaf children
        branch_leaf_info = if_0.val if isinstance(if_0, Leaf) else None
        if_1 = tree_from_bitstring(bit_iter, without, leaf_info=branch_leaf_info)
        return Branch(feat, if_0, if_1)

# compression of a numpy uint8 array
def compress(arr, feat_addr, feat_prev):
    init, feats, labels = get_samples(arr, feat_addr, feat_prev)
    data = list(init)
    if feats.shape[0]:
        tree = make_tree(feats, labels)
    else:
        # not storing a tree when the length is zero makes for
        # several annoying edge cases (ex. what if --size != 0?
        # how will decompress work?) so we just store a single leaf
        tree = Leaf(0)
    data += list(tree.to_bitstring(range(feats.shape[1])))
    data += [0] * (-len(data) % 8)
    data = np.array(data).reshape((-1, 8))
    data = np.array(unbits(data), dtype=np.uint8)
    return tree, data

# util function for decompression
def get_addr_feat(val, size):
    return [(val >> i) & 1 for i in range(size)]

# decompression of a numpy uint8 array
def decompress(arr, feat_addr, feat_prev, size, depth=False):
    arr_bits = bits(arr, 8).flatten()
    poss = range(feat_addr + feat_prev)
    bit_iter = iter(arr_bits[feat_prev:])
    tree = tree_from_bitstring(bit_iter, poss)
    if 8 * size <= feat_prev:
        # we are only reading (a part of) the initial state, so no need
        # to do things like initialize the state or traverse the tree
        return tree, arr[:size], np.zeros(size)
    state = list(reversed(arr_bits[:feat_prev]))
    # reversed, since most recent for feat_prev is 0
    output_bits = np.zeros(8 * size, dtype=np.uint8)
    if not depth: output_bits[:feat_prev] = arr_bits[:feat_prev]
    for i in range(feat_prev, output_bits.size):
        feats = get_addr_feat(i, feat_addr) + state
        next, num_nodes = run_tree_on(tree, feats)
        if depth:
            output_bits[i] = num_nodes
        else:
            output_bits[i] = next
        state = [next] + state[:-1]
    output = output_bits.reshape((-1, 8))
    if depth:
        output = np.array(output.sum(axis=1), dtype=np.uint8)
    else:
        output = np.array(unbits(output), dtype=np.uint8)
    return tree, output

# file format info
header_fmt = "<8sQHBB12x"
header_size = struct.calcsize(header_fmt)
magic = "TREEZIP\x07"

# compression and decompression using command line arguments
# manages all the writing to/from file/numpy array 

def error(msg):
    sys.stderr.write("error: {}\n".format(msg))
    sys.exit(1)

def assert_clean(b, msg):
    if not b: error(msg)

def compress_args(args):
    data_str = args.infile.read()
    if args.feat_addr is None: # "common" case
        feat_addr = int_log2(8 * len(data_str))
    else:
        feat_addr = args.feat_addr
    feat_prev = args.feat_prev
    if args.show_meta:
        sys.stderr.write("feat_addr={}\n".format(feat_addr))
        sys.stderr.write("feat_prev={}\n".format(feat_prev))
    try:
        tree, comp_data = compress(
            np.fromstring(data_str, dtype=np.uint8),
            feat_addr, feat_prev)
    except AmbiguousFeaturesException:
        error("ambiguous features")
    if args.show_meta:
        sys.stderr.write("tree.leaf_count()={}\n".format(tree.leaf_count()))
    if args.show_tree:
        tree.pprint(sys.stderr)
    if not args.suppress:
        args.outfile.write(struct.pack(header_fmt,
            magic,
            len(data_str),
            1,
            feat_addr,
            feat_prev,
        ))
        args.outfile.write(comp_data.tostring())

def decompress_args(args):
    header = args.infile.read(header_size)
    assert_clean(len(header) == header_size,
        "input file is too short")
    header_data = struct.unpack_from(header_fmt, header)
    assert_clean(header_data[0] == magic,
        "input file does not have magic number")
    if args.size is None: # "common" case
        size = header_data[1]
    else:
        size = args.size
    assert_clean(header_data[2] == 1, # version 1
        "version is not equal to 1")
    feat_addr = header_data[3]
    feat_prev = header_data[4]
    if args.show_meta:
        sys.stderr.write("feat_addr={}\n".format(feat_addr))
        sys.stderr.write("feat_prev={}\n".format(feat_prev))
    comp_data = args.infile.read()
    if args.suppress: size = 0
    # this is so we don't try to decompress the full data when
    # supress is enabled, but we still want the tree
    tree, data = decompress(
        np.fromstring(comp_data, dtype=np.uint8),
        feat_addr, feat_prev, size, depth=args.depth)
    if args.show_meta:
        sys.stderr.write("tree.leaf_count()={}\n".format(tree.leaf_count()))
    if args.show_tree:
        tree.pprint(sys.stderr)
    if not args.suppress:
        args.outfile.write(data.tostring())

def get_args():
    parser = argparse.ArgumentParser(
        description="Experimental compression algorithm using decision trees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # controls the "flow" of data
    parser.add_argument("-d", dest="decompress",
        action="store_true", default=False,
        help="decompress instead of compress")
    parser.add_argument("-m", dest="show_meta",
        action="store_true", default=False,
        help="print metadata to stderr")
    parser.add_argument("-t", dest="show_tree",
        action="store_true", default=False,
        help="print the tree to stderr")
    parser.add_argument("-s", dest="suppress",
        action="store_true", default=False,
        help="suppress output (useful when just viewing tree or metadata)")
    # features
    parser.add_argument("-a", dest="feat_addr", metavar="num",
        action="store", type=int, default=None,
        help="number of features that are address bits (default max)")
    parser.add_argument("-p", dest="feat_prev", metavar="num",
        action="store", type=int, default=0,
        help="number of features that are previous bits (default 0)")
    # other stuff
    parser.add_argument("--size", dest="size", metavar="num",
        action="store", type=int, default=None,
        help="how many bytes to decompress (you can even 'extend' the file!)")
    parser.add_argument("--depth", dest="depth",
        action="store_true", default=False,
        help="instead of decompressed data, show max depth reached per byte")
    # data
    parser.add_argument("infile", metavar="infile", nargs="?",
        type=argparse.FileType('rb'), default=sys.stdin,
        help="the input file, or stdin if omitted")
    parser.add_argument("outfile", metavar="outfile", nargs="?",
        type=argparse.FileType('wb'), default=sys.stdout,
        help="the output file, or stdout if omitted")
    # parse the args
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.decompress:
        decompress_args(args)
    else:
        compress_args(args)

if __name__ == "__main__":
    main()
