"""
Tests for the RMSD metrics.
"""

import pytest
from sampleworks.metrics.rmsd import AllAtomRMSD


# TODO: make these a bit more rigorous, similarly to tests/metrics/test_lddt_metrics.py


def test_all_atom_rmsd_identity(altlocA_backbone):
    """A stack compared to itself must have zero RMSD on every residue."""
    metric = AllAtomRMSD()
    results = metric.compute(altlocA_backbone, altlocA_backbone)

    assert results["best_of_1_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert results["residue_rmsd_scores"]
    for residue, scores in results["residue_rmsd_scores"].items():
        assert scores == pytest.approx([0.0], abs=1e-6), f"nonzero identity RMSD at {residue}"


def test_all_atom_rmsd_end_to_end(altlocA_backbone, altlocB_backbone):
    selection_string = "res_id > 179 and res_id < 190"
    metric = AllAtomRMSD()
    results = metric.compute(altlocA_backbone, altlocB_backbone, selection_string)

    expected_results = {
        "best_of_1_rmsd": 0.7332,
        "residue_rmsd_scores": {
            "A180": [2.9206],
            "A181": [4.2650],
            "A182": [5.6639],
            "A183": [2.9397],
            "A184": [2.0595],
            "A185": [2.2885],
            "A186": [3.6929],
            "A187": [1.8138],
            "A188": [0.9547],
            "A189": [0.7596],
        },
    }

    assert results["best_of_1_rmsd"] == pytest.approx(expected_results["best_of_1_rmsd"], abs=0.002)

    assert set(results["residue_rmsd_scores"].keys()) == set(
        expected_results["residue_rmsd_scores"].keys()
    )

    for residue_key, expected_scores in expected_results["residue_rmsd_scores"].items():
        result_scores = results["residue_rmsd_scores"][residue_key]
        assert len(result_scores) == len(expected_scores)
        for got, want in zip(result_scores, expected_scores):
            assert got == pytest.approx(want, abs=0.005), (
                f"RMSD mismatch at {residue_key}: got {got}, expected {want}"
            )


def test_all_atom_rmsd_selection_does_not_change_global(altlocA_backbone, altlocB_backbone):
    """Selection filters the residue-level dict but must not change the global RMSD."""
    metric = AllAtomRMSD()
    all_residues = metric.compute(altlocA_backbone, altlocB_backbone)
    selected = metric.compute(altlocA_backbone, altlocB_backbone, "res_id > 179 and res_id < 190")

    assert selected["best_of_1_rmsd"] == pytest.approx(all_residues["best_of_1_rmsd"], abs=1e-6)
    assert len(selected["residue_rmsd_scores"]) < len(all_residues["residue_rmsd_scores"])
