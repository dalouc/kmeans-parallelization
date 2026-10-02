"""Lab 1, part three: parallel k-means over ``proteins.csv`` with threads.

Same data-parallel (SPMD) decomposition as the multiprocessing version: the
dataset is split into one chunk per thread, every thread runs the same map
step over its own chunk, and the main thread reduces the partial results into
the new centroids. Threads share the address space, so the chunks are passed
straight to them with no copy; the speedup comes from the NumPy operations,
which release the GIL while they run.

Run it from the directory that holds ``proteins.csv``::

    python lab1-proteins-th.py
"""

from __future__ import annotations

import os
import threading
import time
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

SEED = 42
DATA_FILE = "proteins.csv"
FEATURES = ["enzyme", "hydrofob"]
K_VALUES = range(1, 11)
MAX_ITERATIONS = 100
TOLERANCE = 1e-4
SCATTER_SAMPLE = 20_000
# One thread per logical core. Measured on a 16-core machine, this is the
# fastest setting: fewer threads leave cores idle, and more only pay for the
# threads created on every iteration without balancing the work any better.
NUM_THREADS = os.cpu_count() or 1

Points = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int32]
Lengths = npt.NDArray[np.integer]
Chunks = list[Points]
# Contribution of one chunk to an iteration: points per cluster, sum of the
# features per cluster, and the inertia of the chunk.
Partial = tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]

# The threads all append to the same list of partial results, which is the
# only variable they share: a lock keeps that critical section safe.
RESULTS_LOCK = threading.Lock()


def read_dataset() -> tuple[Points, Lengths]:
    """Read the proteins file.

    Only the columns the lab needs are parsed, and the sequences are turned
    into their lengths right away so that the strings can be released.

    Returns:
        The ``(enzyme, hydrofob)`` points and the length of every sequence.
    """
    frame = pd.read_csv(
        DATA_FILE,
        usecols=[*FEATURES, "sequence"],
        dtype={"enzyme": np.float32, "hydrofob": np.float32},
    )
    points: Points = frame[FEATURES].to_numpy(np.float32)
    lengths: Lengths = frame["sequence"].str.len().to_numpy(np.int64)
    return points, lengths


def standardize(points: Points) -> tuple[Points, Points, Points]:
    """Rescale every feature to zero mean and unit standard deviation.

    k-means compares Euclidean distances, so a feature with a wider range
    weighs more in the result: ``hydrofob`` spans 40 to 220 while ``enzyme``
    spans 1 to 10, and without this the clusters would be decided by
    ``hydrofob`` almost on its own.

    Args:
        points: One row per protein, in the original units.

    Returns:
        The standardized points, and the mean and the standard deviation of
        every feature, which :func:`restore` needs to undo the scaling. Both
        are accumulated in double precision, as the dataset holds millions of
        rows.
    """
    mean = points.mean(axis=0, dtype=np.float64).astype(np.float32)
    deviation = points.std(axis=0, dtype=np.float64)
    # A constant feature separates nothing: leave its column at zero instead of
    # dividing by zero.
    deviation = np.where(deviation > 0.0, deviation, 1.0).astype(np.float32)
    return (points - mean) / deviation, mean, deviation


def restore(centroids: Points, mean: Points, deviation: Points) -> Points:
    """Undo :func:`standardize`, to report the centroids in the real units.

    Args:
        centroids: One row per cluster, in standardized units.
        mean: Mean of every feature, as returned by :func:`standardize`.
        deviation: Standard deviation of every feature, likewise.

    Returns:
        The centroids in the original units of the features.
    """
    return centroids * deviation + mean


def assign(points: Points, centroids: Points) -> tuple[Labels, float]:
    """Assign every point to its closest centroid (Euclidean distance).

    Distances are compared one centroid at a time, which keeps the memory in
    use proportional to the number of points instead of points times
    clusters.

    Args:
        points: One row per protein.
        centroids: One row per cluster.

    Returns:
        The cluster index of every point and the inertia (the sum of the
        squared distances to the assigned centroid).
    """
    labels = np.zeros(len(points), np.int32)
    best = np.full(len(points), np.inf, np.float32)
    for cluster, centroid in enumerate(centroids):
        distances = (points[:, 0] - centroid[0]) ** 2 + (
            points[:, 1] - centroid[1]
        ) ** 2
        closer = distances < best
        best[closer] = distances[closer]
        labels[closer] = cluster
    return labels, float(best.sum(dtype=np.float64))


def split(points: Points, parts: int) -> Chunks:
    """Split ``points`` into ``parts`` chunks of almost equal length.

    The chunks are NumPy views, so no protein is copied: the threads read
    them straight from the memory the main thread already holds.

    Args:
        points: One row per protein.
        parts: Number of chunks, one per thread.

    Returns:
        One chunk of the dataset per thread.
    """
    bounds = [round(part * len(points) / parts) for part in range(parts + 1)]
    return [points[start:stop] for start, stop in pairwise(bounds)]


def chunk_step(points: Points, centroids: Points, results: list[Partial]) -> None:
    """Map step: collect what one chunk contributes to the next iteration.

    This is what every thread runs: the counts, the feature sums and the
    inertia of its own chunk are appended to the shared ``results`` list.

    Args:
        points: The chunk of the dataset this thread works on.
        centroids: Current centroids, the same for every thread (SPMD).
        results: Shared list where the partial result is collected.
    """
    labels, inertia = assign(points, centroids)
    k = len(centroids)
    counts = np.bincount(labels, minlength=k)
    totals = np.empty((k, points.shape[1]))
    for feature in range(points.shape[1]):
        totals[:, feature] = np.bincount(
            labels, weights=points[:, feature], minlength=k
        )
    with RESULTS_LOCK:
        results.append((counts, totals, inertia))


def combine(results: list[Partial], centroids: Points) -> tuple[Points, float]:
    """Reduce step: turn the partial results into the moved centroids.

    Args:
        results: One partial result per chunk.
        centroids: Current centroids, kept in place for empty clusters.

    Returns:
        The new centroids and the inertia of the current ones.
    """
    counts = np.sum([count for count, _, _ in results], axis=0)
    totals = np.sum([total for _, total, _ in results], axis=0)
    inertia = sum(value for _, _, value in results)
    nonempty = counts[:, None] > 0
    moved = np.where(nonempty, totals / np.maximum(counts, 1)[:, None], centroids)
    return moved.astype(np.float32), inertia


def parallel_step(chunks: Chunks, centroids: Points) -> list[Partial]:
    """Run one map step, with one thread per chunk.

    Args:
        chunks: One chunk of the dataset per thread.
        centroids: Current centroids, the same for every thread (SPMD).

    Returns:
        One partial result per chunk, in the order the threads finished.
    """
    results: list[Partial] = []
    threads: list[threading.Thread] = []
    for num_thread, chunk in enumerate(chunks):
        thread = threading.Thread(
            name=f"th{num_thread}",
            target=chunk_step,
            args=(chunk, centroids, results),
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return results


def kmeans(
    chunks: Chunks, points: Points, k: int, rng: np.random.Generator
) -> tuple[Points, float]:
    """Cluster ``points`` into ``k`` groups with Lloyd's algorithm.

    Every iteration is a map-reduce round: the threads run :func:`chunk_step`
    over their own chunk and the main thread combines the partial results.

    Args:
        chunks: One chunk of the dataset per thread.
        points: One row per protein, used to draw the initial centroids.
        k: Number of clusters.
        rng: Random generator used to pick the initial centroids.

    Returns:
        The centroids and their inertia.
    """
    centroids = points[rng.choice(len(points), k, replace=False)]
    inertia = 0.0
    for _ in range(MAX_ITERATIONS):
        results = parallel_step(chunks, centroids)
        moved, inertia = combine(results, centroids)
        if np.abs(moved - centroids).max() <= TOLERANCE:
            break
        centroids = moved
    return centroids, inertia


def elbow(
    chunks: Chunks, points: Points, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Compute the inertia of every candidate ``k`` in :data:`K_VALUES`.

    Args:
        chunks: One chunk of the dataset per thread.
        points: One row per protein, used to draw the initial centroids.
        rng: Random generator used to pick the initial centroids.

    Returns:
        The inertia of each candidate ``k``, in the order of :data:`K_VALUES`.
    """
    return np.array([kmeans(chunks, points, k, rng)[1] for k in K_VALUES])


def optimal_k(inertias: npt.NDArray[np.float64]) -> int:
    """Pick the knee of the elbow curve.

    Both axes are scaled to ``[0, 1]`` first, so the chosen ``k`` does not
    depend on the units of the inertia; the answer is the point that lies
    furthest from the straight line joining the two ends of the curve.

    Args:
        inertias: The inertia of each candidate ``k``.

    Returns:
        The optimum number of clusters.
    """
    ks = np.array(K_VALUES, np.float64)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertias - inertias[-1]) / (inertias[0] - inertias[-1])
    # Distance from (x, y) to the line through (0, 1) and (1, 0), i.e. x + y = 1.
    distances = np.abs(x + y - 1) / np.sqrt(2)
    return int(ks[distances.argmax()])


def longest_sequence_cluster(
    labels: Labels, lengths: Lengths, centroids: Points
) -> tuple[int, int, float]:
    """Find the cluster holding the longest sequence.

    Ties are broken by the ``hydrofob`` value of the centroid, as the lab
    material asks.

    Args:
        labels: Cluster index of every point.
        lengths: Length of every sequence.
        centroids: One row per cluster.

    Returns:
        The cluster index, the longest sequence length in the dataset and the
        average sequence length of that cluster.
    """
    longest = int(lengths.max())
    candidates = np.unique(labels[lengths == longest])
    hydrofob = FEATURES.index("hydrofob")
    cluster = int(candidates[centroids[candidates, hydrofob].argmax()])
    average = float(lengths[labels == cluster].mean())
    return cluster, longest, average


def elbow_construction(
    inertias: npt.NDArray[np.float64], k: int
) -> tuple[float, float]:
    """Locate the foot of the perpendicular that :func:`optimal_k` measures.

    :func:`optimal_k` picks the ``k`` lying furthest from the straight line
    joining the two ends of the curve, with both axes scaled to ``[0, 1]``.
    This projects that point onto the line and maps the result back to the
    units of the graph, so :func:`plot_elbow` can draw the distance that was
    actually measured.

    Args:
        inertias: The inertia of each candidate ``k``.
        k: The chosen number of clusters.

    Returns:
        The ``(k, inertia)`` coordinates of the foot of the perpendicular, in
        the units of the plotted axes.
    """
    first, last = K_VALUES.start, K_VALUES[-1]
    lowest, highest = float(inertias[-1]), float(inertias[0])
    x = (k - first) / (last - first)
    y = (float(inertias[k - first]) - lowest) / (highest - lowest)
    # Foot of the perpendicular from (x, y) to the line x + y = 1.
    offset = (x + y - 1) / 2
    return (
        first + (x - offset) * (last - first),
        lowest + (y - offset) * (highest - lowest),
    )


def plot_elbow(inertias: npt.NDArray[np.float64], k: int) -> None:
    """Plot the elbow graph, marking the chosen ``k``.

    Args:
        inertias: The inertia of each candidate ``k``.
        k: The chosen number of clusters.
    """
    optimum = float(inertias[k - K_VALUES.start])
    foot_k, foot_inertia = elbow_construction(inertias, k)
    plt.figure("Elbow graph")
    plt.plot(K_VALUES, inertias, marker="o", label="inertia")
    plt.plot(
        [K_VALUES.start, K_VALUES[-1]],
        [inertias[0], inertias[-1]],
        linestyle="--",
        color="gray",
        label="line joining the ends of the curve",
    )
    plt.plot(
        [k, foot_k],
        [optimum, foot_inertia],
        linestyle=":",
        color="red",
        label="distance from the curve to that line",
    )
    plt.plot(
        k,
        optimum,
        marker="o",
        color="red",
        markersize=12,
        label=f"optimum k = {k}",
    )
    plt.title(f"Elbow graph (optimum k = {k})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (standardized features)")
    plt.xticks(list(K_VALUES))
    plt.grid(visible=True, alpha=0.3)
    plt.legend()
    # The perpendicular is computed with both axes scaled to [0, 1], so it only
    # shows as a right angle if the box it is drawn in is square.
    plt.gca().set_box_aspect(1)


def plot_clusters(
    points: Points, labels: Labels, centroids: Points, rng: np.random.Generator
) -> None:
    """Scatter the clusters and their centroids.

    A random sample of the points is drawn: the full dataset holds millions of
    proteins over a handful of distinct values, so plotting all of them costs
    a lot and shows exactly the same picture.

    Args:
        points: One row per protein.
        labels: Cluster index of every point.
        centroids: One row per cluster.
        rng: Random generator used to draw the sample of points.
    """
    sample = rng.choice(len(points), min(SCATTER_SAMPLE, len(points)), replace=False)
    plt.figure("Clusters")
    plt.scatter(
        points[sample, 0], points[sample, 1], c=labels[sample], cmap="tab10", s=5
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="X",
        s=150,
        label="centroids",
    )
    plt.title(f"Clusters by enzyme and hydrofob (k = {len(centroids)})")
    plt.xlabel("enzyme")
    plt.ylabel("hydrofob")
    plt.legend()


def plot_heatmap(centroids: Points, scaled: Points) -> None:
    """Plot a heat map of the centroid values.

    The color is the standardized value of the feature, which is what k-means
    actually compared, so both features share one scale; the cells are
    annotated with the value in the original units.

    Args:
        centroids: One row per cluster, in the original units.
        scaled: The same centroids, in standardized units.
    """
    extreme = max(float(np.abs(scaled).max()), 1e-12)
    plt.figure("Centroids heat map")
    plt.imshow(scaled, cmap="coolwarm", aspect="auto", vmin=-extreme, vmax=extreme)
    plt.colorbar(label="standard deviations from the mean")
    plt.title("Heat map of the clusters' centroids")
    plt.xticks(range(len(FEATURES)), FEATURES)
    plt.yticks(range(len(centroids)), [f"cluster {i}" for i in range(len(centroids))])
    for cluster, centroid in enumerate(centroids):
        for feature, value in enumerate(centroid):
            plt.text(feature, cluster, f"{value:.2f}", ha="center", va="center")


def main() -> None:
    """Run part three of the lab and print its results."""
    start = time.perf_counter()

    rng = np.random.default_rng(SEED)
    points, lengths = read_dataset()
    # k-means compares Euclidean distances, so it runs on the standardized
    # features; the centroids come back to the original units to be reported.
    scaled, mean, deviation = standardize(points)
    chunks = split(scaled, NUM_THREADS)
    inertias = elbow(chunks, scaled, rng)
    k = optimal_k(inertias)
    scaled_centroids, _ = kmeans(chunks, scaled, k, rng)
    labels, _ = assign(scaled, scaled_centroids)
    centroids = restore(scaled_centroids, mean, deviation)
    cluster, longest, average = longest_sequence_cluster(labels, lengths, centroids)

    elapsed = time.perf_counter() - start

    print(f"Number of threads: {NUM_THREADS}")
    print(f"Proteins read: {len(points)}")
    print(f"Optimum number of clusters (k): {k}")
    print("Cluster with the highest sequence length:")
    print(f"  id: {cluster}")
    print(f"  highest sequence length: {longest}")
    print(f"  average sequence length: {average:.2f}")
    print(
        f"  centroid: enzyme={centroids[cluster, 0]:.2f}, "
        f"hydrofob={centroids[cluster, 1]:.2f}"
    )
    print(f"Execution time: {elapsed:.3f} s")

    plot_elbow(inertias, k)
    plot_clusters(points, labels, centroids, rng)
    plot_heatmap(centroids, scaled_centroids)
    plt.show()


if __name__ == "__main__":
    main()
