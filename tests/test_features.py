from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoswitch.data import load_library  # noqa: E402
from thermoswitch.features import (  # noqa: E402
    DINUCS,
    TRIN,
    build_feature_matrix,
    dinuc_freqs,
    gc_fraction,
    get_feature_names,
    rbs_window_onehot,
    sequence_length,
    trinuc_freqs,
)


def test_gc_fraction():
    assert gc_fraction("AUGC") == 0.5


def test_sequence_length():
    assert sequence_length("AUGC") == 4
    assert sequence_length("") == 0


def test_dinuc_feature_vector_length_is_16():
    assert len(dinuc_freqs("AUGC")) == 16


def test_trinuc_feature_vector_length_is_64():
    assert len(trinuc_freqs("AUGC")) == 64


def test_kmer_frequencies_for_valid_and_short_sequences():
    dinuc = dinuc_freqs("AUG")
    assert np.isclose(dinuc[DINUCS.index("AU")], 0.5)
    assert np.isclose(dinuc[DINUCS.index("UG")], 0.5)
    assert np.isclose(dinuc.sum(), 1.0)
    assert np.allclose(dinuc_freqs("A"), np.zeros(16))

    trinuc = trinuc_freqs("AUGC")
    assert np.isclose(trinuc[TRIN.index("AUG")], 0.5)
    assert np.isclose(trinuc[TRIN.index("UGC")], 0.5)
    assert np.isclose(trinuc.sum(), 1.0)
    assert np.allclose(trinuc_freqs("AU"), np.zeros(64))


def test_rbs_window_onehot_output_length_is_fixed_at_144():
    vec = rbs_window_onehot("AUGCAUGCAUGCAUGC", 4, 10)

    assert len(vec) == 144


def test_missing_rbs_coordinates_return_zero_vector_of_length_144():
    vec = rbs_window_onehot("AUGCAUGCAUGCAUGC", None, None)

    assert len(vec) == 144
    assert np.allclose(vec, np.zeros(144))


def test_build_feature_matrix_returns_expected_number_of_rows():
    df = load_library().head(3)
    matrix = build_feature_matrix(df)

    assert matrix.shape[0] == 3


def test_get_feature_names_length_matches_feature_matrix_width():
    df = load_library().head(2)
    matrix = build_feature_matrix(df)

    assert len(get_feature_names()) == matrix.shape[1]


def test_feature_names_do_not_contain_target_or_output_columns():
    forbidden = {
        "closed_low",
        "open_midhi",
        "open_high",
        "T50",
        "slope70",
        "slope_max",
        "reentrant_pen",
        "score",
        "curve",
        "p_pass",
        "t50_pred",
        "slope70_pred",
        "rank",
    }

    for name in get_feature_names():
        assert name not in forbidden
