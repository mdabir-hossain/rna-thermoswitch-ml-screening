from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoswitch.shortlist import (  # noqa: E402
    gc_fraction_for_shortlist,
    has_homopolymer,
    jaccard_distance,
    kmer_set,
    make_order_sheet,
    passes_prediction_guards,
    passes_qc,
    rank_key,
    refine_shortlist,
    select_top_k,
)


def row(row_id, seq="AUGCAUGCAUGC", p_pass=0.5, t50_pred=70.0, slope70_pred=0.05):
    return {
        "id": row_id,
        "seq": seq,
        "p_pass": p_pass,
        "t50_pred": t50_pred,
        "slope70_pred": slope70_pred,
    }


def test_rank_key_sorts_higher_p_pass_first():
    rows = [row("low", p_pass=0.2), row("high", p_pass=0.9)]

    ranked = sorted(rows, key=rank_key, reverse=True)

    assert ranked[0]["id"] == "high"


def test_t50_closeness_breaks_ties():
    rows = [row("far", t50_pred=72.0), row("near", t50_pred=70.1)]

    ranked = sorted(rows, key=rank_key, reverse=True)

    assert ranked[0]["id"] == "near"


def test_kmer_set_returns_expected_3mers():
    assert kmer_set("AUGC", 3) == {"AUG", "UGC"}


def test_jaccard_distance_identical_sequences_is_zero():
    assert jaccard_distance("AUGCAU", "AUGCAU") == 0.0


def test_jaccard_distance_is_higher_for_different_sequences():
    same = jaccard_distance("AUGCAU", "AUGCAU")
    different = jaccard_distance("AUGCAU", "CCCCCC")

    assert different > same


def test_has_homopolymer_detects_7_repeated_bases():
    assert has_homopolymer("AAAAAAAU") is True


def test_passes_qc_fails_low_gc():
    assert passes_qc(row("x", seq="AUAUAUAUAU")) is False


def test_passes_qc_fails_high_gc():
    assert passes_qc(row("x", seq="GCGCGCGCGC")) is False


def test_passes_qc_fails_homopolymer():
    assert passes_qc(row("x", seq="AAAAAAAGCGC")) is False


def test_passes_qc_passes_acceptable_sequence():
    assert passes_qc(row("x", seq="AUGCAUGCAUGC")) is True
    assert gc_fraction_for_shortlist("AUGCAUGCAUGC") == pytest.approx(0.5)


def test_passes_prediction_guards_respects_ranges():
    assert passes_prediction_guards(row("ok", t50_pred=70.0, slope70_pred=0.05)) is True
    assert passes_prediction_guards(row("bad_t50", t50_pred=73.0, slope70_pred=0.05)) is False
    assert passes_prediction_guards(row("bad_slope", t50_pred=70.0, slope70_pred=0.03)) is False


def test_select_top_k_returns_expected_length_on_controlled_rows():
    rows = [row(str(i), p_pass=1.0 - i / 10, t50_pred=70.0, slope70_pred=0.05) for i in range(5)]

    selected = select_top_k(rows, k=3, use_guards=True)

    assert len(selected) == 3
    assert [item["rank"] for item in selected] == [1, 2, 3]


def test_refine_shortlist_replaces_failing_candidates_when_possible():
    predictions = [
        row("bad", seq="AAAAAAAAAA", p_pass=0.99, t50_pred=70.0, slope70_pred=0.05),
        row("good1", seq="AUGCAUGCAUGC", p_pass=0.90, t50_pred=70.0, slope70_pred=0.05),
        row("good2", seq="UGCAUGCAUGCA", p_pass=0.80, t50_pred=70.0, slope70_pred=0.05),
    ]
    shortlist = [
        {"rank": 1, "id": "bad", "seq": "AAAAAAAAAA"},
        {"rank": 2, "id": "good1", "seq": "AUGCAUGCAUGC"},
    ]

    refined = refine_shortlist(predictions, shortlist, k=2)

    assert len(refined) == 2
    assert "bad" not in {item["id"] for item in refined}
    assert "good2" in {item["id"] for item in refined}


def test_make_order_sheet_returns_required_columns():
    shortlist = [
        {
            "rank": 1,
            "id": "a",
            "seq": "AUGCAUGCAUGC",
            "p_pass": 0.9,
            "t50_pred": 70.0,
            "slope70_pred": 0.05,
        }
    ]

    order_rows = make_order_sheet(shortlist)

    assert set(order_rows[0]) == {
        "rank",
        "id",
        "seq",
        "len_nt",
        "gc_frac",
        "p_pass",
        "t50_pred",
        "slope70_pred",
    }
