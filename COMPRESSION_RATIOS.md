# Example compression ratios

This page lists some compression ratios of example files, as well as some other interesting examples. Additionally, I compare the values to those of `gzip`, a standard compression tool.

(Actually, these are inverse compression ratios, to emphasize the relative sizes of before/after files).

For an explanation of why increasing the number of features sometimes increases the length of the file, see [FORMAT_INFO.md](FORMAT_INFO.md).

## Examples

### `treezip.py`

Even compressing `treezip.py` itself, the file size increases by about a factor of 1.5. (The exact value depends on the value of `-p`.)

| Value for `-p` | Change in size |
| --- | --- |
| 0 | 1.781 |
| 8 | 1.804 |
| 16 | 1.428 |
| 24 | 1.337 |
| 32 | 1.394 |
| Use `gzip` instead | 0.321 |

### `ls`

The same holds for compressing the Linux `ls` binary, which one would expect match many of the examples given.

| Value for `-p` | Change in size |
| --- | --- |
| 0 | 1.720 |
| 8 | 1.798 |
| 16 | 1.680 |
| 24 | 1.699 |
| 32 | 1.831 |
| Use `gzip` instead | 0.454 |

### `thue_morse.bin`

This is a file of length 4K, whose bits are exactly the [Thue-Morse sequence](https://en.wikipedia.org/wiki/Thue–Morse_sequence) (0110100110010110...). It was used to demonstrate the worst-case behavior of the decision tree with only address features. Since the bit value is the parity of the address, the decision tree can't represent the data except as a full binary tree. It turns out that just adding one previous bit can drastically improve the size of the decision tree, though.

| Value for `-p` | Change in size (exact size) |
| --- | --- | 
| 0 | 3.324 (13616) |
| 1 | 0.013 (55) |
| Use `gzip` instead | 0.027 (110) |

So at least it works well for something?

### `head -c 4096 /dev/null`

Clearly, TreeZip should perform extremely well on this (since it just needs to store that all bits are 0).

| Value for `-p` | Change in size (exact size) |
| --- | --- | 
| 0 | 0.008 (33) |
| Use `gzip` instead | 0.005 (20) |

Looks like the header is too big. (Note: the new sizes are the same for 65536).

### `yes | head -c 65536`

| Value for `-p` | Change in size (exact size) |
| --- | --- | 
| 0 | 0.001 (41) |
| 8 | 0.001 (41) |
| 16 | 0.001 (36) |
| Use `gzip` instead | 0.002 (99) |

The reason `-p 16` has such a drastic ("drastic") decrease is that there, TreeZip just needs to store the first two characters, and a very small decision tree that says to copy the bit exactly two bytes back.

### `for x in {1..100}; do echo "The quick brown fox jumps over the lazy dog"; done`

(length of each line is 44 bytes)

| Value for `-p` | Change in size (exact size) |
| --- | --- | 
| 0 | 1.650 (7260) |
| 8 | 0.340 (1494) |
| 16 | 0.061 (270) |
| 32 | 0.039 (172) |
| 64 | 0.042 (185) |
| Use `gzip` instead | 0.020 (89) |

The reason for the poor performance with `-p 0` is because the data is not aligned to a power of 2. (Regular compression algorithms deal with this by refering to data a particular distance before the current location.)

### `for x in {1..100}; do echo "The quick brown fox jumps over the lazy dog                    "; done`

(length of each line is 64 bytes)

*There should be 20 trailing space characters in the string, but it seems Markdown eats multiple spaces in inline code (see <https://talk.commonmark.org/t/multiple-spaces-in-backticks/1088>).*

| Value for `-p` | Change in size (exact size) |
| --- | --- | 
| 0 | 0.027 (175) |
| 8 | 0.028 (176) |
| 16 | 0.031 (196) |
| 32 | 0.029 (187) |
| 64 | 0.036 (231) |
| Use `gzip` instead | 0.016 (101) |

Here, TreeZip was able to use the alignment to its benefit.

## Summary

TreeZip only compresses files which are extremely repetitive, and have their patterns aligned well to powers of two. Additionally, the compression ratio depends significantly on the value of `-p` (the number of previous bits used as features).

