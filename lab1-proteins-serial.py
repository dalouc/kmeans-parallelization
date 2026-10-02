"""Lab 1, part one: serial k-means over ``proteins.csv``.

The program reads the dataset, standardizes the features so that both weigh
the same in the Euclidean distance, builds the elbow graph to choose the
optimum number of clusters, clusters the proteins by ``enzyme`` and
``hydrofob``, reports the cluster holding the longest sequence and prints the
execution time. Every figure is built only after the clock has been stopped, so nothing
blocks while the work being measured is running.

Run it from the directory that holds ``proteins.csv``::

    python lab1-proteins-serial.py
"""

from __future__ import annotations

import time

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

Points = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int32]
Lengths = npt.NDArray[np.integer]


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


def update(points: Points, labels: Labels, centroids: Points) -> Points:
    """Move every centroid to the mean of the points assigned to it.

    Args:
        points: One row per protein.
        labels: Cluster index of every point.
        centroids: Current centroids, kept in place for empty clusters.

    Returns:
        The new centroids.
    """
    k = len(centroids)
    counts = np.bincount(labels, minlength=k)
    moved = np.empty_like(centroids)
    for feature in range(points.shape[1]):
        totals = np.bincount(labels, weights=points[:, feature], minlength=k)
        moved[:, feature] = np.where(
            counts > 0, totals / np.maximum(counts, 1), centroids[:, feature]
        )
    return moved


def kmeans(
    points: Points, k: int, rng: np.random.Generator
) -> tuple[Points, Labels, float]:
    """Cluster ``points`` into ``k`` groups with Lloyd's algorithm.

    Args:
        points: One row per protein.
        k: Number of clusters.
        rng: Random generator used to pick the initial centroids.

    Returns:
        The centroids, the cluster index of every point and the inertia.
    """
    centroids = points[rng.choice(len(points), k, replace=False)]
    labels, inertia = assign(points, centroids)
    for _ in range(MAX_ITERATIONS):
        moved = update(points, labels, centroids)
        if np.abs(moved - centroids).max() <= TOLERANCE:
            break
        centroids = moved
        labels, inertia = assign(points, centroids)
    return centroids, labels, inertia


def elbow(points: Points, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Compute the inertia of every candidate ``k`` in :data:`K_VALUES`.

    Args:
        points: One row per protein.
        rng: Random generator used to pick the initial centroids.

    Returns:
        The inertia of each candidate ``k``, in the order of :data:`K_VALUES`.
    """
    return np.array([kmeans(points, k, rng)[2] for k in K_VALUES])


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
    """Run part one of the lab and print its results."""
    start = time.perf_counter()

    rng = np.random.default_rng(SEED)
    points, lengths = read_dataset()
    # k-means compares Euclidean distances, so it runs on the standardized
    # features; the centroids come back to the original units to be reported.
    scaled, mean, deviation = standardize(points)
    inertias = elbow(scaled, rng)
    k = optimal_k(inertias)
    scaled_centroids, labels, _ = kmeans(scaled, k, rng)
    centroids = restore(scaled_centroids, mean, deviation)
    cluster, longest, average = longest_sequence_cluster(labels, lengths, centroids)

    elapsed = time.perf_counter() - start

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
