"""
Bag-of-words tests.

Place recognition must rank the SAME place above a different one. These use
synthetic descriptor sets so the expected answer is known rather than eyeballed.
"""

from __future__ import annotations

import numpy as np

from vslam.bow import assign_words, bow_vector, build_vocabulary, fit_idf


def _descriptors(rng, n: int, base: np.ndarray | None = None) -> np.ndarray:
    if base is None:
        return rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    # A noisy re-observation of the same place: same descriptors, perturbed.
    noise = rng.integers(-8, 9, size=base.shape)
    return np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)


def test_vocabulary_declines_on_too_little_data() -> None:
    rng = np.random.default_rng(0)
    assert build_vocabulary([_descriptors(rng, 10)], size=256) is None


def test_same_place_scores_higher_than_a_different_one() -> None:
    rng = np.random.default_rng(0)
    place_a = _descriptors(rng, 400)
    place_b = _descriptors(rng, 400)
    revisit_a = _descriptors(rng, 400, base=place_a)

    vocabulary = build_vocabulary([place_a, place_b, revisit_a], size=32)
    assert vocabulary is not None
    fit_idf(vocabulary, [place_a, place_b, revisit_a])

    va = bow_vector(vocabulary, place_a)
    vb = bow_vector(vocabulary, place_b)
    vr = bow_vector(vocabulary, revisit_a)

    # The revisit must look more like place A than place B does.
    assert float(va @ vr) > float(va @ vb)


def test_vectors_are_normalised() -> None:
    rng = np.random.default_rng(1)
    sets = [_descriptors(rng, 300) for _ in range(3)]
    vocabulary = build_vocabulary(sets, size=32)
    assert vocabulary is not None
    for descriptors in sets:
        assert abs(np.linalg.norm(bow_vector(vocabulary, descriptors)) - 1.0) < 1e-5


def test_empty_descriptors_are_handled() -> None:
    rng = np.random.default_rng(2)
    vocabulary = build_vocabulary([_descriptors(rng, 300)], size=32)
    assert vocabulary is not None
    assert len(assign_words(vocabulary, np.empty((0, 32), dtype=np.uint8))) == 0
    assert np.allclose(bow_vector(vocabulary, None), 0.0)
