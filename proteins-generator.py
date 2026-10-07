"""Generate `proteins.csv`, the synthetic dataset the lab's k-means programs read.

Usage:
    python proteins-generator.py <lines> <seed>

Each row is `protid,enzyme,hydrofob,sequence`: a 1-based id, an EC number in
1..10, a hydrophobicity drawn from a range that the EC number selects, and a
residue string over the eight letters A..H. Roughly one row in eight gets a
known motif prefixed to its sequence.

For any given `<lines> <seed>` the output is byte for byte what the original
course generator produced. That is the correctness contract, and it is what
makes the rewrite below safe: the file is regenerated from a seed rather than
committed, so a changed byte would silently change every published measurement.

Why it is written this way
--------------------------
Every draw the generator needs is `random._randbelow(n)` for some small `n`:
`randrange(1, 256, 8)` is `1 + 8 * below(32)`, `randint(a, b)` is
`a + below(b - a + 1)`, and `choice` over the eight residues is `below(8)`.
For `n < 256`, `below` asks Mersenne Twister for `getrandbits(k)` with
`k = n.bit_length() <= 8`, which burns one whole 32-bit word and keeps only its
top `k` bits, retrying while the result lands outside `[0, n)`.

So the whole program is a walk along one sequence of 32-bit words that never
looks past the top byte of each. `randbytes(4 * m)` draws `m` words at once and
lays them down little-endian, so `[3::4]` of those bytes is exactly the top
byte of every word. Working a batch at a time turns the hot path into plain
integer compares on a `bytes`, and lets the residues of a whole sequence be
sliced out pre-translated instead of drawn one letter at a time: a sequence is
just the next run of words whose top byte is < 128, so `positions` and
`residues` (below) resolve it in two indexing operations. Rows are formatted
into a `bytearray` and written out in multi-megabyte blocks.
"""

# S311: a reproducible pseudo-random dataset is the entire point of this
#       script, and reproducing the original's bytes means reproducing the
#       exact Mersenne Twister stream. There is no crypto here to get wrong.
# ruff: noqa: S311

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import BinaryIO

# Words drawn from Mersenne Twister per batch, and bytes buffered per write.
# The per-row cost is independent of the batch size; this only trades how often
# a batch is refilled against how much memory one holds.
WORDS_PER_BATCH = 1 << 18
FLUSH_BYTES = 1 << 22

# A draw of `below(n)` keeps a word whose top byte `t` satisfies
# `t >> (8 - n.bit_length()) < n`. Every bound the generator uses is >= 128, so
# a word that gets rejected always has `t >= 128` -- which is what lets the
# residue run and the rejected words share one predicate (see ACCEPTED below).
#
# `below(8)`, the residue draw:      keep t < 128, residue = "A" + (t >> 4)
# `below(32)`, the sequence length:  keep t < 128, length  = 1 + 8 * (t >> 2)
# `below(10)`, the EC number:        keep t < 160, enzyme  = 1 + (t >> 4)
# `below(7)`, the motif test:        keep t < 224, value   = 1 + (t >> 5)
ACCEPTED = 128
ENZYME_BOUND = 160
MOTIF_BOUND = 224

# The `randint` range `hydrofob` is drawn from, as (base, bound, shift), for
# each EC number 1..10 -- randint(40, 75) below 3, randint(55, 140) below 6,
# randint(125, 175) below 8, randint(160, 220) above. Indexing this by the EC
# number replaces the original's chain of comparisons. `bound` is the accept
# test pre-multiplied so that it applies directly to the top byte.
_LOW = (40, 36 << 2, 2)
_MID = (55, 86 << 1, 1)
_HIGH = (125, 51 << 2, 2)
_TOP = (160, 61 << 2, 2)
HYDROFOB = (None, _LOW, _LOW, _MID, _MID, _MID, _HIGH, _HIGH, _TOP, _TOP, _TOP)

# Maps the top byte of an accepted word to its residue letter, "A" + (t >> 4).
# Bytes >= 128 cannot occur in `accepted` but are masked so the table is total.
RESIDUE = bytes(ord("A") + ((top & 0x7F) >> 4) for top in range(256))

# A flagged row prefixes "ABCD" to its sequence, plus "CDEFGH" on every second
# flagged row and a further "ABCD" on every fourth. Indexed by motif count % 4.
MOTIFS = (b"ABCDCDEFGHABCD", b"ABCD", b"ABCDCDEFGH", b"ABCD")

HEADER = b"protid,enzyme,hydrofob,sequence\n"
ROW = b"%d,%d,%d,%s%s\n"


def refill(rng: random.Random, carry: bytes) -> tuple[bytes, bytes, bytes, np.ndarray]:
    """Draw a fresh batch of words, appended to the `carry` tail of the old one.

    Returns the top byte of each word, the top bytes of just the accepted
    words, those same bytes translated to residue letters, and the position of
    each accepted word within the batch.
    """
    tops = carry + rng.randbytes(4 * WORDS_PER_BATCH)[3::4]
    words = np.frombuffer(tops, np.uint8)
    positions = np.flatnonzero(words < ACCEPTED)
    accepted = words[positions].tobytes()
    return tops, accepted, accepted.translate(RESIDUE), positions


def generate(handle: BinaryIO, rng: random.Random, lines: int) -> None:
    """Write `lines` rows to `handle`, drawing from `rng`."""
    tops, accepted, residues, positions = refill(rng, b"")
    where = positions.item

    # `cursor` indexes `tops`; `taken` counts the accepted words before it, so
    # it indexes `accepted` / `residues`. `motifs` counts the flagged rows.
    cursor = taken = motifs = 0
    out = bytearray()

    protid = 1
    while protid <= lines:
        # A batch runs dry mid-row sooner or later. Reading past its end raises
        # IndexError, and since every draw is a pure function of the word
        # sequence, replaying the row over the leftover tail plus a fresh batch
        # reproduces it exactly. This costs nothing per row and happens once
        # per WORDS_PER_BATCH words.
        start, start_motifs = cursor, motifs
        try:
            # The sequence: its length comes from the next accepted word, and
            # its residues are the run of accepted words after that.
            last = taken + 1 + (accepted[taken] >> 2) * 8
            cursor = where(last) + 1
            sequence = residues[taken + 1 : last + 1]
            taken = last + 1

            # The EC number. Each draw below walks `cursor` past the words it
            # rejects and leaves `top` holding the one it keeps.
            while (top := tops[cursor]) >= ENZYME_BOUND:
                cursor += 1
            cursor += 1
            taken += top < ACCEPTED
            enzyme = 1 + (top >> 4)

            # Hydrophobicity, over the range this EC number selects.
            base, bound, shift = HYDROFOB[enzyme]
            while (top := tops[cursor]) >= bound:
                cursor += 1
            cursor += 1
            taken += top < ACCEPTED
            hydrofob = base + (top >> shift)

            # Flag this row if the draw matches its position in the cycle.
            while (top := tops[cursor]) >= MOTIF_BOUND:
                cursor += 1
            cursor += 1
            taken += top < ACCEPTED
            motif = b""
            if 1 + (top >> 5) == (protid - 1) % 8:
                motifs += 1
                motif = MOTIFS[motifs % 4]
        except IndexError:
            cursor, taken, motifs = 0, 0, start_motifs
            tops, accepted, residues, positions = refill(rng, tops[start:])
            where = positions.item
            continue

        out += ROW % (protid, enzyme, hydrofob, motif, sequence)
        if len(out) >= FLUSH_BYTES:
            handle.write(out)
            out.clear()
        protid += 1

    handle.write(out)


def main() -> None:
    lines = int(sys.argv[1])
    seed = int(sys.argv[2])
    # The original echoed its line count; `sys.stdout.write` rather than
    # `print` to mark that this is the script's output, not a leftover debug.
    sys.stdout.write(f"{sys.argv[1]}\n")

    rng = random.Random(seed)
    with Path("proteins.csv").open("wb") as handle:
        handle.write(HEADER)
        generate(handle, rng, lines)


if __name__ == "__main__":
    main()
