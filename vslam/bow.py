"""
Bag-of-words place recognition, for detecting when the camera returns somewhere.

THE PROBLEM

Loop closure needs to answer "have I seen this before?" against every keyframe
in the map. Matching descriptors pairwise against every keyframe is quadratic in
sequence length and far too slow.

A bag-of-words index makes it a vector comparison. Each keyframe is summarised as
a histogram over a vocabulary of descriptor clusters -- which visual words it
contains and how often -- so two places that look alike produce similar
histograms, and a query is a dot product instead of a descriptor match.

WHY NOT DBoW2

DBoW2 is the standard choice and what ORB-SLAM uses. Its Python bindings are
awkward to build, and a pretrained vocabulary is a large binary blob. Since the
vocabulary here is built from the sequence being processed, it is trained on
exactly the right distribution anyway.

WHY cv2.kmeans AND NOT scikit-learn

Clustering ORB descriptors needs k-means, and scikit-learn's MiniBatchKMeans is
the obvious tool. OpenCV already ships k-means, and this project's whole argument
is about what fits in a constrained compute budget -- adding a large dependency
for one function call would sit badly with that. No accuracy claim is made for
one over the other; this is a dependency decision, not a numerical one.

A CAVEAT WORTH STATING

ORB descriptors are BINARY, and k-means on binary data treated as floats is not
the theoretically correct clustering -- DBoW2 uses k-majority, which works in
Hamming space. Cluster centres here are real-valued and a descriptor is assigned
by Euclidean distance. It works well enough for place recognition, where only the
relative ordering of similarities matters, but it is an approximation and is
labelled as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_VOCABULARY_SIZE = 256
# Descriptors sampled to train the vocabulary. All of them would be slower with
# no measurable benefit, since k-means converges on a representative sample.
MAX_TRAINING_DESCRIPTORS = 40000


@dataclass
class Vocabulary:
    """Cluster centres plus the inverse-document-frequency weight per word."""

    centres: np.ndarray            # (K, 32) float32
    idf: np.ndarray                # (K,) float32

    @property
    def size(self) -> int:
        return len(self.centres)


def build_vocabulary(
    descriptor_sets: list[np.ndarray],
    size: int = DEFAULT_VOCABULARY_SIZE,
    seed: int = 0,
) -> Vocabulary | None:
    """
    Cluster descriptors from every keyframe into a visual vocabulary.

    Returns None when there is too little data to cluster, which is an ordinary
    outcome on a short sequence rather than an error.
    """
    usable = [d for d in descriptor_sets if d is not None and len(d)]
    if not usable:
        return None

    pool = np.vstack(usable).astype(np.float32)
    if len(pool) < size * 2:
        return None

    rng = np.random.default_rng(seed)
    if len(pool) > MAX_TRAINING_DESCRIPTORS:
        pool = pool[rng.choice(len(pool), MAX_TRAINING_DESCRIPTORS, replace=False)]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centres = cv2.kmeans(
        pool, size, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # idf is filled in by `fit_idf` once every keyframe has been assigned; a
    # uniform weight is the sensible starting point.
    return Vocabulary(centres=centres, idf=np.ones(size, dtype=np.float32))


def assign_words(vocabulary: Vocabulary, descriptors: np.ndarray) -> np.ndarray:
    """Nearest cluster centre for each descriptor."""
    if descriptors is None or len(descriptors) == 0:
        return np.empty(0, dtype=int)
    d = descriptors.astype(np.float32)
    # (N, K) distances via the expansion |a-b|^2 = |a|^2 - 2a.b + |b|^2, which
    # avoids materialising an (N, K, 32) difference array.
    distances = (
        (d ** 2).sum(axis=1, keepdims=True)
        - 2.0 * d @ vocabulary.centres.T
        + (vocabulary.centres ** 2).sum(axis=1)[None, :]
    )
    return np.argmin(distances, axis=1)


def bow_vector(vocabulary: Vocabulary, descriptors: np.ndarray) -> np.ndarray:
    """
    L2-normalised, idf-weighted word histogram for one keyframe.

    L2 normalisation makes the comparison a cosine similarity, so a keyframe with
    more features is not automatically more similar to everything.
    """
    words = assign_words(vocabulary, descriptors)
    vector = np.zeros(vocabulary.size, dtype=np.float32)
    if len(words) == 0:
        return vector
    np.add.at(vector, words, 1.0)
    vector *= vocabulary.idf
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def fit_idf(vocabulary: Vocabulary, descriptor_sets: list[np.ndarray]) -> None:
    """
    Weight words by how rare they are, in place.

    A word appearing in every keyframe carries no information about WHICH place
    this is -- it is the visual equivalent of "the". Inverse document frequency
    downweights those and lets distinctive structure dominate the similarity.
    """
    document_count = np.zeros(vocabulary.size, dtype=np.float64)
    n_documents = 0
    for descriptors in descriptor_sets:
        if descriptors is None or len(descriptors) == 0:
            continue
        n_documents += 1
        document_count[np.unique(assign_words(vocabulary, descriptors))] += 1.0
    if n_documents == 0:
        return
    # +1 inside the log keeps a word present in every document at weight 0
    # rather than negative, and the outer +1 avoids dividing by zero.
    vocabulary.idf = np.log(
        (n_documents + 1.0) / (document_count + 1.0)
    ).astype(np.float32)
