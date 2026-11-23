from pathlib import Path

from ips_learning.visualization.utils import (
    StudyFigureRepository,
    StudyTableRepository,
)


def test_study_figure_repository_path(tmp_path: Path) -> None:
    repo = StudyFigureRepository(root_dir=tmp_path)

    path = repo.stage_dir(
        study_name="scalability",
        stage_name="summary",
    )

    assert path == (
        tmp_path
        / "studies"
        / "scalability"
        / "figures"
        / "summary"
    )
    assert path.is_dir()


def test_study_table_repository_path(tmp_path: Path) -> None:
    repo = StudyTableRepository(root_dir=tmp_path)

    path = repo.stage_dir(
        study_name="scalability",
        stage_name="summary",
    )

    assert path == (
        tmp_path
        / "studies"
        / "scalability"
        / "tables"
        / "summary"
    )
    assert path.is_dir()
