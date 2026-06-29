import pytest
import sys
from pathlib import Path

import test
sys.path.append(str(Path(__file__).parent.parent))

from src.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


def _make_encoder() -> RandomDistributedScalarEncoder:
    params = RDSEParameters(
        size=256,
        active_bits=16,
        sparsity=0.0,
        radius=0.0,
        resolution=0.05,
        category=False,
        seed=42,
    )
    return RandomDistributedScalarEncoder(params)


def _make_large_encoder(radius: float = 1.0) -> RandomDistributedScalarEncoder:
    params = RDSEParameters(
        size=10000,
        sparsity=0.02,
        radius=radius,
        seed=12345,
    )
    return RandomDistributedScalarEncoder(params)


def _overlap_count(first: list[int], second: list[int]) -> int:
    return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)


def test_decode_closest_prefers_highest_overlap():
    encoder = _make_encoder()
    for value in (0.0, 0.5, 1.0):
        encoder.register_encoding(value)

    query = encoder.encode(0.52)
    decoded_value, confidence = encoder.decode(query)

    assert decoded_value == pytest.approx(0.5)
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.5


def test_decode_closest_works_with_candidates_only():
    encoder = _make_encoder()
    query = encoder.encode(1.0)

    decoded_value, confidence = encoder.decode(query, candidates=[0.0, 1.0])

    assert decoded_value == pytest.approx(1.0)
    assert confidence >= 0.9


def test_clear_registered_encodings_resets_cache():
    encoder = _make_encoder()
    encoder.register_encoding(0.25)
    # Decoding against a populated cache resolves to the registered value.
    sdr = encoder._compute_encoding(0.25)
    assert encoder.decode(sdr)[0] == pytest.approx(0.25)

    encoder.clear_registered_encodings()
    # With the cache cleared and no candidates, decode has nothing to match.
    # (encode() must NOT be used here: it re-registers the value via register_encoding.)
    value, confidence = encoder.decode(sdr)
    assert value is None
    assert confidence == 0.0


def test_rdse_overlap_within_radius_large_encoding():
    # w = 0.02 * 10000 = 200 active bits; separation 0.4 (< radius 1.0) gives
    # structural overlap ~ w * (1 - 0.4/radius) = 120.
    encoder = _make_large_encoder(radius=1.0)
    values = [i * 0.1 for i in range(200)]
    for value in values:
        within = value + 0.4
        overlap = _overlap_count(encoder.encode(value), encoder.encode(within))
        assert overlap > 100


def test_rdse_no_overlap_outside_radius_large_encoding():
    # Separation 5.0 (>> radius 1.0) gives no structural overlap; only the random
    # collision floor (w^2/n = 4) remains, so overlap stays well below 0.1*w = 20.
    encoder = _make_large_encoder(radius=1.0)
    values = [i for i in range(200)]
    for value in values:
        outside = value + 5.0
        overlap = _overlap_count(encoder.encode(value), encoder.encode(outside))
        assert overlap < 20


def test_rdse_encodings_are_mostly_orthogonal():
    encoder = _make_large_encoder(radius=1.0)

    test_baseline_sdr = [0]*(int(encoder.size-encoder.sparsity*encoder.size)) + [1]*int(encoder.sparsity*encoder.size)
    test_baseline_sdrs = []
    import random
    for i in range(6000):
        random.shuffle(test_baseline_sdr)
        test_baseline_sdrs.append(test_baseline_sdr.copy())
    firsts = test_baseline_sdrs[:3000]
    seconds = test_baseline_sdrs[3000:]
    overlaps = []
    for i,j in zip(firsts, seconds):
        overlaps.append(_overlap_count(i, j))

    test_orthogonal_ratio = sum(1 for overlap in overlaps if overlap <= 2) / len(overlaps)
    test_mean_overlap = sum(overlaps) / len(overlaps)

    print(f"Baseline Orthogonal ratio: {test_orthogonal_ratio:.3f}, Mean overlap: {test_mean_overlap:.3f}")

    values = [random.randint(0, 100000) for i in range(3000)]
    encodings = [encoder.encode(value) for value in values]

    firsts = random.choices(range(len(values)), k=3000)
    seconds = random.choices(range(len(values)), k=3000)
    overlaps = []
    for i,j in zip(firsts, seconds):
        first = encodings[i]
        second = encodings[j]
        overlaps.append(_overlap_count(first, second))

    orthogonal_ratio = sum(1 for overlap in overlaps if overlap <= 2) / len(overlaps)
    mean_overlap = sum(overlaps) / len(overlaps)

    print(f"Orthogonal ratio: {orthogonal_ratio:.3f}, Mean overlap: {mean_overlap:.3f}")
    # test within range of baseline
    assert orthogonal_ratio >= test_orthogonal_ratio - .05
    assert mean_overlap <= test_mean_overlap + .5
