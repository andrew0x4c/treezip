# TreeZip format

This document details the format of TreeZip-compressed files.


## Header

The TreeZip header consists of 32 bytes. All multi-byte fields are stored little-endian.

### (8) Magic number (`magic`)

The TreeZip magic number is `0x54 0x52 0x45 0x45 0x5a 0x49 0x50 0x07`, or as a string, `"TREEZIP\x07"`. The `\x07` is just to ensure that the terminal beeps if you `cat` a TreeZip-compressed file.

### (8) Size (`size`)

The size of the uncompressed file. Yes, this means that TreeZip can theoretically store 18446744073709551615-byte (16 EiB) files. However, since TreeZip currently loads the data into RAM to compress and decompress, this is very unlikely to be achieved.

### (2) Version (`version`)

The version of the TreeZip format used to compress the file. This is currently 1, and will likely remain 1 for quite a while.

### (1) Features which are address bits (`feat_addr`)

The number of address bits used as features. By default, this is always enough bits to address each bit in the original file (`ceil(log2(# bits in file))`); however, it is possible to reduce this manually if the file is repetitive enough or `feat_prev` is high enough.

### (1) Features which are previous bits (`feat_prev`)

The number of previously output bits used as features. By default, this is 0; however, it may be useful to increase this depending on the data.

### (12) Padding

12 bytes of padding, for a total of 32 bytes. These bytes are currently ignored. This allows for future expansion of the TreeZip format, as well as making a TreeZip-compressed file look nice in `xxd`. (Also, 32 bytes is 256 bits, which is a nice round number.)


## Data

After here, there is no alignment (not even per byte). This is to ensure that the data is packed efficiently.

Note that TreeZip always considers the first (0th) bit of a byte to be the least significant bit.

Let `num_feats` denote the total number of features (`feat_addr + feat_prev`).

### Initialization

To allow the decompression to start, the first `feat_prev` bits of the original file are written directly to the compressed file.

If the number of bits in the file is less than or equal to `feat_prev`, the initialization bits are padded with 0's, and the tree stored is just `Leaf(0)`.

### Features

In a decision tree, a feature is represented as an integer from 0 to `num_feats-1`. The first `feat_addr` features denote the corresponding bit of the current bit's address (with LSB first), and the next `feat_prev` features denote the previous bits (starting from the most recently output bit and going backwards).

### Tree

Let `Leaf(b)` denote a leaf which returns the value `b`.

Let `Branch(f, x, y)` denote a branch which goes to subtree `x` if feature `f` is 0, and subtree `y` if feature `f` is 1.

To encode a tree, the following algorithm is used:

    EncodeFeat(F, f) = F.index(f), from LSB to MSB, encoded with ceil(log2(len(F))) bits (exactly enough bits to represent all elements of F unambiguously)
    EncodeTree(F, Leaf(b)) = 0 b
    EncodeTree(F, Branch(f, Leaf(b0), Leaf(b1))) = 1 EncodeFeat(F, f) 0 b0 0
    EncodeTree(F, Branch(f, x, y)) = 1 EncodeFeat(F, f) EncodeTree(F without f, x) EncodeTree(F without f, y)

To encode the entire tree, `EncodeTree([0, 1, ... num_feats-1], T)` is called.

The justifications behind the design choices are as following:

1. The reason the first bit selects between branch/leaf is because there will always be a 50/50 split (well, there's actually one extra leaf, but this is not significant).
2. The second case of `EncodeTree` deals with the following property: in a correctly constructed decision tree, we will never have a branch of the form `Branch(f, Leaf(b), Leaf(b))`, as this could always be replaced with `Leaf(b)`. Thus, if we have encoded `1 EncodeFeat(F, f) 0 b 0` (that is, `Branch(f, Leaf(b), Leaf(...`), we know that the other leaf must be the opposite of `b`, and thus, don't record it.
3. The reason we juggle around the list of unused features `F`, is because of the following property: in a correctly constructed decision tree, we will never first use a feature, then use it again in one of the subtrees. Thus, we just don't bother with representing the already used features. (This change actually gave a somewhat significant improvement in the amount of compressions.)

It is fairly straightforward to see that this encoding is reversible.

Finally, to make sure the number of bits in the file reaches an integral byte, we pad with 0 bits (though the actual bits used are ignored).
