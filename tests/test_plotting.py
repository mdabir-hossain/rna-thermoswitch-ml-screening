from pathlib import Path

from thermoswitch.plotting import EXPECTED_FIGURE_FILENAMES, make_all_figures


def test_make_all_figures_creates_expected_files(tmp_path: Path) -> None:
    paths = make_all_figures(output_dir=tmp_path)
    names = {path.name for path in paths}

    assert set(EXPECTED_FIGURE_FILENAMES) == names
    for filename in EXPECTED_FIGURE_FILENAMES:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
