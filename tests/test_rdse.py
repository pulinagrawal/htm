import pytest

from rdse import RandomDistributedScalarEncoder, RDSEParameters


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
    encoder.clear_registered_encodings()

    with pytest.raises(ValueError):
        encoder.decode(encoder.encode(0.25))
