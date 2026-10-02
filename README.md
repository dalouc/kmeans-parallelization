# Lab 1 — k-means parallelization in Python

Master in Big Data — *Technological Fundamentals in the Big Data World*.

Parts one to three are implemented: the serial program and its two parallel
versions, with multiprocessing and with threads. Part four, the written
report, is not part of this repository.

| Part | Program                   | Parallelism               |
| ---- | ------------------------- | ------------------------- |
| 1    | `lab1-proteins-serial.py` | none                      |
| 2    | `lab1-proteins-mp.py`     | `multiprocessing`, `Pool` |
| 3    | `lab1-proteins-th.py`     | `threading`, `Thread`     |

The three programs print exactly the same results — same seed, same initial
centroids, same arithmetic — and differ only in how the work is spread. That
was checked across the eighteen worker configurations measured below: all of
them printed the same `k`, the same cluster and the same averages.

## Dataset

The dataset is produced by `proteins-generator.py`, which is course material
and is kept byte for byte as it was handed out (the lint and format hooks skip
it). It writes `proteins.csv` into the current directory:

```bash
python proteins-generator.py 50000 42      # development
python proteins-generator.py 2000000 42    # performance measurements
```

`proteins.csv` is not committed.

## Running

```bash
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
python -m pip install --group run  # --group dev also installs the tooling
python lab1-proteins-serial.py     # part one
python lab1-proteins-mp.py         # part two
python lab1-proteins-th.py         # part three
```

The `run` group is the three libraries the programs import; `dev` adds ruff,
pyright and pre-commit, and is what the [everyday
commands](#everyday-commands) need.

The lab forbids absolute or relative paths to the dataset, so the programs
read `proteins.csv` from the working directory: run them from the directory
that holds the file.

LAB1.pdf spells the threaded program `lab1-Proteins-th.py`, with a capital
`P`. The file here is all lowercase: rename it before zipping the delivery if
the exact spelling is graded.

## What the programs do

1. Starts the clock.
2. Reads `proteins.csv`, keeping `enzyme`, `hydrofob` and the length of every
   `sequence`.
3. Standardizes `enzyme` and `hydrofob`, so that both weigh the same in the
   euclidean distance.
4. Runs k-means for `k = 1..10` and builds the elbow curve.
5. Picks the optimum `k` as the point of the curve furthest from the straight
   line joining its two ends, with both axes scaled to `[0, 1]` first.
6. Clusters the proteins with that `k`, and brings the centroids back to the
   original units to report them.
7. Finds the cluster holding the longest sequence — ties broken by the
   `hydrofob` of the centroid — and averages the sequence lengths in it.
8. Stops the clock, prints the results and the execution time.
9. Only then builds the three figures and shows them: elbow graph, clusters
   with their centroids, and a heat map of the centroid values.

Nothing blocks while the measured work is running, as the lab requires: no
figure is created before the clock has been stopped.

k-means is implemented from scratch on top of NumPy: euclidean distance,
random initial centroids drawn from the data, and Lloyd's iteration until the
centroids move less than `TOLERANCE` or `MAX_ITERATIONS` is reached. The
random generator is always seeded with `SEED = 42`, so runs are reproducible.

### Standardizing the features

Euclidean distance gives the wider feature the louder vote. On the
2,000,000-protein dataset `hydrofob` has mean 127.8 and standard deviation
53.5, against 5.50 and 2.87 for `enzyme`: on the raw columns a one-unit step
in `enzyme` is noise next to the spread of `hydrofob`, so the clusters are cut
along `hydrofob` almost on its own.

Each column is therefore centred on its mean and divided by its standard
deviation, both accumulated in double precision over the millions of rows.
The whole run happens in that space — elbow curve included, since the inertia
of a candidate `k` has to be comparable to the others — and the centroids are
multiplied back by the standard deviation and shifted by the mean before being
printed and plotted, so every reported value is in `enzyme` and `hydrofob`
units. The round trip is exact, and the scaling itself costs 12 ms of the 20 s
run.

This changes the answer rather than just tidying it up:

| Features     | Optimum `k` | Centroids (`enzyme`, `hydrofob`)              |
| ------------ | ----------- | --------------------------------------------- |
| raw columns  | 3           | (2.56, 65.12), (5.12, 128.37), (8.50, 185.23) |
| standardized | 2           | (2.98, 81.02), (7.97, 173.70)                 |

The generator ties the two columns together — it draws `hydrofob` from a range
chosen by the enzyme number — so the honest reading of the standardized result
is that the dataset holds two groups, low-`enzyme`/low-`hydrofob` and
high-`enzyme`/high-`hydrofob`, and that the third cluster the raw columns found
was a slice of the `hydrofob` axis rather than a group of proteins.

The centroids heat map colours the standardized value, which is what the
algorithm actually compared, and annotates each cell with the real one. The
elbow graph's inertia axis is in standardized units too, and says so.

## How the parallel versions work

Both follow the same data-parallel (SPMD) decomposition, applied to the inner
loop of k-means, which is where all the time goes:

- **Decomposition.** The dataset is split once into one chunk per worker.
- **Map.** Every worker runs the same step over its own chunk: assign its
  points to the closest centroid and reduce them to three small partial
  results — points per cluster, sum of the features per cluster, and inertia.
- **Reduce.** The master adds the partial results up and moves the centroids.
  Only the centroids travel out and only the partial sums come back, so the
  data stays where it is for the whole run.

`lab1-proteins-mp.py` creates one `mp.Pool(mp.cpu_count())`, whose
`initializer` hands the dataset to every process once, and then calls
`pool.starmap` per iteration over `CHUNKS_PER_PROCESS` chunks per worker.

`lab1-proteins-th.py` starts `os.cpu_count()` `threading.Thread`s per
iteration; they share the address space, so each one receives its chunk as a
NumPy view with no copy, and they collect their partial results in a shared
list protected by a lock. NumPy releases the GIL while it works, which is what
makes the threaded version gain anything at all.

## Measured times

Median of three interleaved runs on 2,000,000 proteins, `SEED = 42`, on an
otherwise idle i5-13500H (12 cores — 4 performance cores with SMT and 8
efficiency cores — 16 logical):

| Version         | Time    | Speedup | Karp-Flatt |
| --------------- | ------- | ------- | ---------- |
| serial          | 20.20 s | 1.00    | —          |
| multiprocessing |  4.72 s | 4.28    | 0.183      |
| threads         |  6.00 s | 3.37    | 0.250      |

Measure on an idle machine, and compare absolute times only within one
measurement session: under sustained load the CPU settles into a lower clock,
and the serial program took 19.04 s as the first run of this session against
20.20 s once the machine had been busy for a few minutes. Speedups are
unaffected.

### Where the time goes

Median of three runs, serial version:

| Phase                       | Time     | Share |
| --------------------------- | -------- | ----- |
| reading the CSV             |  1.825 s |  8.9% |
| standardizing               |  0.012 s |  0.1% |
| elbow curve, `k = 1..10`    | 18.419 s | 90.0% |
| final k-means and reporting |  0.213 s |  1.0% |

Only the k-means steps are parallelized, so 9.0% of the work stays serial and
Amdahl caps the speedup at 11.1× however many workers are added, and at 6.8×
with the 16 this machine has. Multiprocessing reaches 4.28× against that
ceiling of 6.8×, and the Karp-Flatt metric puts the experimentally determined
serial fraction at 0.183 against the 0.090 actually measured: the gap between
the two is parallel overhead — the per-iteration dispatch, and memory
bandwidth, since the k-means steps move far more data than they compute.

Standardizing shortened the run rather than lengthening it. The scaling itself
costs 12 ms, but Lloyd's iteration converges sooner on the scaled columns:
building the elbow curve takes 142 iterations instead of 242, and the serial
version measured back to back, same session, drops from 36.33 s to 20.12 s.
The whole gain lands on the parallelizable part, which is why the speedups in
the table above are lower than they were before the change: the 1.8 s CSV read
is unchanged and now a larger share of a smaller total. The programs do the
same amount of useful work in less time and scale slightly worse while doing
it.

### Choosing the number of workers

Both counts were measured rather than guessed, three runs per setting on the
2,000,000-protein dataset, interleaved and reported as medians.

One caveat for reading the spreads below: the clock drifts down as the machine
heats, and a setting measured later in a sweep pays for it. The same
configuration, measured at three positions of the same sweep, gave 4.72 s,
4.90 s and 5.06 s — so differences under about 7% are drift, not signal.

Processes, at four chunks per process: 7.87 s with 4, 5.95 s with 8, 5.09 s
with 12, 4.90 s with 16. One process per logical core it is — the step from 12
to 16 is inside the noise, but it is free, and past 16 the machine has nothing
left to give.

Chunks per process, with 16 processes: 5.75 s with one chunk each, 5.08 s with
two, 5.06 s with four, 4.97 s with six, 4.94 s with eight, 5.04 s with twelve.
One chunk each is the only setting that is clearly worse, by 12%; everything
from two to twelve is one flat optimum. The cores are not equal — four
performance cores and eight efficiency ones — so one equally sized chunk per
process makes every iteration wait for the slowest core. Cutting the data
finer lets the pool hand the extra chunks to whoever is free first, and
`CHUNKS_PER_PROCESS = 4` sits in the middle of the flat part.

Threads: 6.87 s with 8, 6.19 s with 16, 6.17 s with 32, 6.98 s with 48, 7.90 s
with 64. One thread per logical core again — 32 ties with 16 and there is no
reason to prefer it. The threaded version cannot use the chunking trick: a
chunk is a thread there, and threads are created and joined on every
iteration, so extra chunks only add overhead instead of balancing the load.
That, plus the parts of NumPy that keep the GIL, is why threads end up slower
than processes.

## Layout

```
.
├── .github/workflows/ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── authors.txt               # one line per author, NIA;SURNAMES;NAME
├── lab1-proteins-mp.py       # part two, multiprocessing
├── lab1-proteins-serial.py   # part one, serial
├── lab1-proteins-th.py       # part three, threads
├── proteins.csv              # generated, not committed
├── proteins-generator.py     # course material, untouched
├── pyproject.toml            # every tool is configured here
└── README.md
```

The delivery expects standalone `.py` files, so each part is a single script
at the root of the repository rather than a package. The three programs
therefore repeat the code they have in common — reading the dataset,
standardizing, choosing the optimum `k`, reporting and plotting — instead of
importing it from a shared module that could not be delivered.

## Everyday commands

| Task                | Command                      |
| ------------------- | ---------------------------- |
| Lint                | `ruff check .`               |
| Lint and autofix    | `ruff check --fix .`         |
| Format              | `ruff format .`              |
| Check formatting    | `ruff format --check .`      |
| Type-check          | `pyright`                    |
| Run every hook      | `pre-commit run --all-files` |
| Update hook pins    | `pre-commit autoupdate`      |
