from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class FigureRepository:
    """
    Consistent saving of figure artifacts.

    Conventions:
        Without profile:
            artifacts/{system_name}/figures/{stage_name}/{file_name}.png

        With profile:
            artifacts/{system_name}/{profile_name}/figures/{stage_name}/{file_name}.png
    """

    def __init__(self, root_dir: str | Path = "artifacts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(
        self,
        *,
        system_name: str,
        stage_name: str,
        profile_name: str | None = None,
    ) -> Path:
        base = self.root_dir / system_name

        if profile_name is not None:
            base = base / profile_name

        path = base / "figures" / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        fig,
        *,
        system_name: str,
        stage_name: str,
        file_name: str,
        profile_name: str | None = None,
        ext: str = "png",
        close: bool = True,
        also_pdf: bool = True,
        dpi: int = 300,
    ) -> Path:
        out_dir = self.stage_dir(
            system_name=system_name,
            stage_name=stage_name,
            profile_name=profile_name,
        )

        path = out_dir / f"{file_name}.{ext}"

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        if also_pdf and ext.lower() != "pdf":
            pdf_path = out_dir / f"{file_name}.pdf"
            fig.savefig(
                pdf_path,
                bbox_inches="tight",
            )

        if close:
            plt.close(fig)

        return path

    def save_gif(
        self,
        ani,
        *,
        system_name: str,
        stage_name: str,
        file_name: str,
        profile_name: str | None = None,
        fps: int = 15,
    ) -> Path:
        out_dir = self.stage_dir(
            system_name=system_name,
            stage_name=stage_name,
            profile_name=profile_name,
        )

        path = out_dir / f"{file_name}.gif"

        ani.save(
            path,
            writer="pillow",
            fps=fps,
        )

        return path


class TableRepository:
    """
    Consistent saving of tabular artifacts.

    Conventions:
        Without profile:
            artifacts/{system_name}/tables/{stage_name}/{file_name}.csv

        With profile:
            artifacts/{system_name}/{profile_name}/tables/{stage_name}/{file_name}.csv
    """

    def __init__(self, root_dir: str | Path = "artifacts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(
        self,
        *,
        system_name: str,
        stage_name: str,
        profile_name: str | None = None,
    ) -> Path:
        base = self.root_dir / system_name

        if profile_name is not None:
            base = base / profile_name

        path = base / "tables" / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_csv(
        self,
        df: pd.DataFrame,
        *,
        system_name: str,
        stage_name: str,
        file_name: str,
        profile_name: str | None = None,
        index: bool = False,
    ) -> Path:
        out_dir = self.stage_dir(
            system_name=system_name,
            stage_name=stage_name,
            profile_name=profile_name,
        )

        path = out_dir / f"{file_name}.csv"
        df.to_csv(path, index=index)
        return path

    def save_tex(
        self,
        df: pd.DataFrame,
        *,
        system_name: str,
        stage_name: str,
        file_name: str,
        profile_name: str | None = None,
        index: bool = False,
        escape: bool = False,
        float_format: str = "%.4e",
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        out_dir = self.stage_dir(
            system_name=system_name,
            stage_name=stage_name,
            profile_name=profile_name,
        )

        path = out_dir / f"{file_name}.tex"

        latex = df.to_latex(
            index=index,
            escape=escape,
            float_format=float_format,
            caption=caption,
            label=label,
        )

        path.write_text(latex, encoding="utf-8")
        return path

    def save(
        self,
        df: pd.DataFrame,
        *,
        system_name: str,
        stage_name: str,
        file_name: str,
        profile_name: str | None = None,
        index: bool = False,
        escape: bool = False,
        float_format: str = "%.4e",
        caption: str | None = None,
        label: str | None = None,
        save_csv: bool = True,
        save_tex: bool = True,
    ) -> dict:
        out = {}

        if save_csv:
            out["csv"] = self.save_csv(
                df=df,
                system_name=system_name,
                stage_name=stage_name,
                file_name=file_name,
                profile_name=profile_name,
                index=index,
            )

        if save_tex:
            out["tex"] = self.save_tex(
                df=df,
                system_name=system_name,
                stage_name=stage_name,
                file_name=file_name,
                profile_name=profile_name,
                index=index,
                escape=escape,
                float_format=float_format,
                caption=caption,
                label=label,
            )

        return out

class StudyFigureRepository:
    """
    Save figures that summarize results from more than one system.

    Convention:
        artifacts/studies/{study_name}/figures/{stage_name}/{file_name}.png
        artifacts/studies/{study_name}/figures/{stage_name}/{file_name}.pdf

    Use FigureRepository for artifacts belonging to one mathematical system.
    """

    def __init__(self, root_dir: str | Path = "artifacts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(
        self,
        *,
        study_name: str,
        stage_name: str,
    ) -> Path:
        path = (
            self.root_dir
            / "studies"
            / study_name
            / "figures"
            / stage_name
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        fig,
        *,
        study_name: str,
        stage_name: str,
        file_name: str,
        ext: str = "png",
        close: bool = True,
        also_pdf: bool = True,
        dpi: int = 300,
    ) -> Path:
        out_dir = self.stage_dir(
            study_name=study_name,
            stage_name=stage_name,
        )

        path = out_dir / f"{file_name}.{ext}"

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        if also_pdf and ext.lower() != "pdf":
            pdf_path = out_dir / f"{file_name}.pdf"
            fig.savefig(
                pdf_path,
                bbox_inches="tight",
            )

        if close:
            plt.close(fig)

        return path

    def save_gif(
        self,
        ani,
        *,
        study_name: str,
        stage_name: str,
        file_name: str,
        fps: int = 15,
    ) -> Path:
        out_dir = self.stage_dir(
            study_name=study_name,
            stage_name=stage_name,
        )

        path = out_dir / f"{file_name}.gif"

        ani.save(
            path,
            writer="pillow",
            fps=fps,
        )

        return path


class StudyTableRepository:
    """
    Save tables that summarize results from more than one system.

    Convention:
        artifacts/studies/{study_name}/tables/{stage_name}/{file_name}.csv
        artifacts/studies/{study_name}/tables/{stage_name}/{file_name}.tex

    Use TableRepository for artifacts belonging to one mathematical system.
    """

    def __init__(self, root_dir: str | Path = "artifacts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(
        self,
        *,
        study_name: str,
        stage_name: str,
    ) -> Path:
        path = (
            self.root_dir
            / "studies"
            / study_name
            / "tables"
            / stage_name
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_csv(
        self,
        df: pd.DataFrame,
        *,
        study_name: str,
        stage_name: str,
        file_name: str,
        index: bool = False,
    ) -> Path:
        out_dir = self.stage_dir(
            study_name=study_name,
            stage_name=stage_name,
        )

        path = out_dir / f"{file_name}.csv"
        df.to_csv(path, index=index)
        return path

    def save_tex(
        self,
        df: pd.DataFrame,
        *,
        study_name: str,
        stage_name: str,
        file_name: str,
        index: bool = False,
        escape: bool = False,
        float_format: str = "%.4e",
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        out_dir = self.stage_dir(
            study_name=study_name,
            stage_name=stage_name,
        )

        path = out_dir / f"{file_name}.tex"

        latex = df.to_latex(
            index=index,
            escape=escape,
            float_format=float_format,
            caption=caption,
            label=label,
        )

        path.write_text(latex, encoding="utf-8")
        return path

    def save(
        self,
        df: pd.DataFrame,
        *,
        study_name: str,
        stage_name: str,
        file_name: str,
        index: bool = False,
        escape: bool = False,
        float_format: str = "%.4e",
        caption: str | None = None,
        label: str | None = None,
        save_csv: bool = True,
        save_tex: bool = True,
    ) -> dict:
        out = {}

        if save_csv:
            out["csv"] = self.save_csv(
                df=df,
                study_name=study_name,
                stage_name=stage_name,
                file_name=file_name,
                index=index,
            )

        if save_tex:
            out["tex"] = self.save_tex(
                df=df,
                study_name=study_name,
                stage_name=stage_name,
                file_name=file_name,
                index=index,
                escape=escape,
                float_format=float_format,
                caption=caption,
                label=label,
            )

        return out
