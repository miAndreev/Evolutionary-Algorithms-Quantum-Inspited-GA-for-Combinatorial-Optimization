import numpy as np
import pytest

from qiga.problems import KnapsackKind
from qiga.problems import generate_euclidean_tsp
from qiga.problems import generate_knapsack
from qiga.representations import BinaryDecoder
from qiga.representations import RandomKeyDecoder
from qiga.representations import make_decoder


def test_binary_decoder_is_identity():
    decoder = BinaryDecoder(4)

    assert decoder.n_bits == 4
    assert decoder.decode(np.array([1, 0, 1, 1], dtype=np.int8)) == [1, 0, 1, 1]
    np.testing.assert_array_equal(decoder.encode([0, 1, 1, 0]), [0, 1, 1, 0])


def test_decoders_reject_wrong_length():
    with pytest.raises(ValueError, match="expected 6 bits"):
        RandomKeyDecoder(3, 2).decode([0, 1])
    with pytest.raises(ValueError, match="positive"):
        BinaryDecoder(0)


@pytest.mark.parametrize("bits_per_key", [0, 31])
def test_random_key_rejects_bad_key_size(bits_per_key):
    with pytest.raises(ValueError, match="bits_per_key"):
        RandomKeyDecoder(5, bits_per_key)


def test_random_keys_are_read_msb_first():
    decoder = RandomKeyDecoder(3, 3)
    bits = [1, 0, 1, 0, 0, 1, 0, 1, 1]  # keys 5, 1, 3

    np.testing.assert_array_equal(decoder.keys(bits), [5, 1, 3])
    assert decoder.decode(bits) == [1, 2, 0]


def test_equal_keys_keep_index_order():
    assert RandomKeyDecoder(4, 2).decode(np.zeros(8, dtype=np.int8)) == [0, 1, 2, 3]


def test_random_bits_always_decode_to_a_permutation():
    rng = np.random.default_rng(0)
    decoder = RandomKeyDecoder(20, 4)

    for _ in range(50):
        assert sorted(decoder.decode(rng.integers(0, 2, decoder.n_bits))) == list(range(20))


@pytest.mark.parametrize(("n", "bits_per_key"), [(2, 1), (5, 3), (16, 4), (30, 8)])
def test_random_key_encode_round_trips(n, bits_per_key):
    decoder = RandomKeyDecoder(n, bits_per_key)
    tour = np.random.default_rng(n).permutation(n).tolist()

    bits = decoder.encode(tour)

    assert bits.shape == (decoder.n_bits,)
    assert decoder.decode(bits) == tour


def test_random_key_encode_validates():
    with pytest.raises(ValueError, match="permutation"):
        RandomKeyDecoder(3, 2).encode([0, 0, 1])
    with pytest.raises(ValueError, match="distinct keys"):
        RandomKeyDecoder(5, 2).encode([0, 1, 2, 3, 4])


@pytest.mark.parametrize(("n", "expected"), [(2, 2), (30, 6), (32, 6), (100, 8)])
def test_default_bits_per_key(n, expected):
    assert RandomKeyDecoder.default_bits_per_key(n) == expected


def test_make_decoder_picks_by_encoding():
    knapsack = generate_knapsack(7, KnapsackKind.UNCORRELATED)
    tsp = generate_euclidean_tsp(10)

    assert isinstance(make_decoder(knapsack), BinaryDecoder)
    assert make_decoder(knapsack, bits_per_key=5).n_bits == 7
    assert make_decoder(tsp).n_bits == 10 * 5
    assert make_decoder(tsp, bits_per_key=3).n_bits == 30
