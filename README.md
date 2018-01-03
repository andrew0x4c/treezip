# TreeZip
General purpose compression algorithm based on decision trees

Not to be confused with the other TreeZip for phylogenetic trees: <http://faculty.cse.tamu.edu/tlw/bmcbioinfo11/> - sorry about the name conflict!

The name originated from some of the other compression formats with the name "[something]ee"-zip (bzip2, gzip).


## Overview

Summary of decision tree induction at Wikipedia: <https://en.wikipedia.org/wiki/Decision_tree_learning>

Given an arbitrary input file, TreeZip tries to learn a decision tree to predict a bit based on its position in the file and the previously output bits. It then writes the tree to a file as the compressed data. To decompress a file, TreeZip uses the decision tree to sequentially compute each bit in the file.

The specific binary-level encoding of a compressed file is given in [FORMAT_INFO.md](FORMAT_INFO.md).

Before actually using TreeZip, you might want to read the section "How good is it?" below.


## Running

Using TreeZip to compress and decompress files is fairly straightforward:
    
    # Basic usage:
    # compress
    [python] treezip.py foo.txt foo.txt.tz
    # decompress
    [python] treezip.py -d foo.txt.tz foo.txt

    # If any are omitted or set to -, stdin and stdout are assumed:
    # compress stdin to foo.tz
    [python] treezip.py - foo.tz
    # decompress stdin to stdout
    [python] treezip.py -d
    
TreeZip has some options which let you tune the compression algorithm:

    # include 16 just-output bits as features (default is 0)
    [python] treezip.py foo.txt foo.txt.tz -p 16
    # same, but ignore all but 4 least significant address bits (default is automatic)
    [python] treezip.py foo.txt foo.txt.tz -a 4 -p 16
    # if you aren't too careful, certain sets of features will be ambiguous!
    # this extreme example will always fail if the input is not empty:
    [python] treezip.py -a 0 -p 0

TreeZip can also visualize various properties of the tree and compression, as well as other interesting features:

    # (s)uppress output and print (m)etadata and (t)ree:
    # uncompressed file
    [python] treezip.py foo.txt -smt
    # compressed file
    [python] treezip.py -d foo.txt.tz -smt

    # when decompressing, output total depth reached per byte, instead of actual data
    [python] treezip.py -d foo.txt.tz --depth

    # use the decision tree to "predict" what comes next in the file
    # (assume size of foo.txt was less than 1000)
    [python] treezip.py -d foo.txt.tz --size 1000


## Why?

The following are some of the thoughts that lead to this project:

1. In a plaintext file, the high bit of each byte is unset. In other words, laying out the bit number from MSB to LSB, any bit whose number is of the form (...xxxx xxxx 111) is always a 0.
2. In binary files, there are often large blocks of zeros. For example, it might be the case that bytes 0x1000 to 0x1800 are all zeros, so we could say that any bit whose number is of the form (...0001 0xxx xxxx xxxx xxx) is always a zero.
3. Similarly, binary files often have repeated bytes, or patterns involving repeated bytes. For example, if there are repeating fields of the form (4 bytes zero, 4 bytes other) in the range 0x1000 to 0x1800, we could say that any bit whose number is of the form (...0001 0xxx xxxx x0xx xxx) is always a zero.

(Note that my bits project provided some of my inspiration for these properties ;)

Because of this, it seems that attempting to find patterns in the bit numbers could be a plausible method of compressing data. For example, if an algorithm encodes the first rule listed above, it "automatically" achieves a compression ratio of about 8/7.

Additionally, often some parts of a file will depend on the bits coming right before it; this was the intent behind adding the previous bits as features. (Note that this means that the decompression cannot usually be parallelized when previous bits are used as features.)

The inspiration to use a machine learning model to compress data partly comes from a remark on [this demo of ConvnetJS](https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html).


## How good is it?

Not very, for several reasons.

### Compression ratio

For most common files I have been able to find, TreeZip actually increases the size of the file (often by about 50%). See [COMPRESSION_RATIOS.md](COMPRESSION_RATIOS.md) for more details.

### Speed

Compression ratio aside, TreeZip takes a very long time to compress a file, because it has to loop over the entire input data, once for each combination of feature and node. For example, the `ls` examples took several minutes to "compress". This also means that TreeZip cannot do streaming compression.


## Potential improvements

- More features
  - ANDs of some of the address bits (ex. so the plaintext case mentioned earlier can be dealt with in one branch)
  - ORs of some of the address bits (for symmetry)
  - XORs of some of the address bits
    - (wait, isn't all this just called 'feature engineering'?)
  - Features which repeat with period of 3 * 2^n, 5 * 2^n, 7 * 2^n? I've noticed that some parts of ELF files repeat with a period of 3 * 2^n for some n.
- More models
  - (*) Rule-based classifier - would be able to learn the above examples in one rule each
  - (*) Naive Bayes
  - Perceptron (learning linear combinations of features)
  - Multi-layer neural network
  - Convolutional neural network (what? maybe 1D ConvNets)
  - Recurrent neural network / LSTM
  - [WaveNet](https://arxiv.org/abs/1609.03499)-style causal convolutions
    - Okay, these last few aren't actually serious
- (*) Denser/more clever encoding of the decision tree
- (*) More clever methods to choose the features, besides greedy algorithm
- (*) Instead of predicting bits, predict bytes or other larger units of data

The ones labeled (*) I feel might give more significant improvements.


## License

Licensed under GPLv3. See the LICENSE file for details.

Copyright (c) Andrew Li 2018

