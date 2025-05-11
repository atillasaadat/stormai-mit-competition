"""
datahandler.py
==============

A small utility for reading *initial-state*, *OMNI2*, and *orbit-mean-density*
CSV files organised in numbered-ID filenames such as:

    omni2-20250601-00042.csv
    sat_density-…-00042-truth.csv

The 5-digit “file ID” is parsed from the stem.  The same ID is used to locate
the matching row in the initial-state table.

Typical usage
-------------

>>> from pathlib import Path
>>> from datahandler import DataHandler
>>> dh = DataHandler(
...     omni2_folder=Path("/data/omni2"),
...     sat_density_folder=Path("/data/sat_density"),
...     initial_state_file=Path("/data/initial_states.csv"),
... )
>>> init = dh.get_initial_state(42)
>>> omni = dh.read_csv_data(42, dh.omni2_folder)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

_LOGGER = logging.getLogger(__name__)         # module-level fallback logger
_ID_PATTERN = re.compile(r"-(\d{5})-")        # matches "-01234-" in filename


class DataHandler:
    """Lightweight façade around CSV I/O for the STORM-AI data layout."""

    # --------------------------------------------------------------------- #
    # construction / initial-state loading
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        omni2_folder: Path,
        sat_density_folder: Path,
        initial_state_file: Path | None = None,
        initial_state_folder: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or _LOGGER
        self.omni2_folder = omni2_folder
        self.sat_density_folder = sat_density_folder

        if initial_state_file and initial_state_folder:
            raise ValueError("Provide *either* initial_state_file or folder, not both.")

        self._initial_state_src = initial_state_file or initial_state_folder
        if self._initial_state_src is None:
            raise ValueError("Must supply initial_state_file or initial_state_folder")

        self.initial_states: pd.DataFrame = self._load_initial_states()
        if self.initial_states.empty:
            raise RuntimeError("Initial-state table is empty — check your path.")

        self.logger.info(
            "DataHandler ready: %d initial states, OMNI2=%s, rho=%s",
            len(self.initial_states), omni2_folder, sat_density_folder
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def get_initial_state(self, file_id: int | str) -> pd.Series:
        """Return the *single* initial-state row for `file_id`."""
        fid = int(file_id)
        row = self.initial_states[self.initial_states["File ID"] == fid]
        if row.empty:
            raise KeyError(f"Initial state for File ID {fid} not found.")
        return row.iloc[0]

    def read_csv_data(self, file_id: int | str, folder: Path) -> pd.DataFrame:
        """
        Locate the CSV in *folder* whose stem contains "-<5-digit ID>-" and
        return it sorted by Timestamp.
        """
        path = self._find_file(int(file_id), folder)
        self.logger.debug("Reading %s", path.name)
        df = (
            pd.read_csv(path, dtype={"Timestamp": "string"})
            .assign(Timestamp=lambda d: pd.to_datetime(
                d["Timestamp"], format="%Y-%m-%d %H:%M:%S", utc=True))
        )
        return df.sort_values("Timestamp", ignore_index=True)

    def save_df_from_copy_folder_path(
        self,
        file_id: int | str,
        df: pd.DataFrame,
        copy_folder: Path,
        dest_folder: Path,
        *,
        return_path_only: bool = False,
    ) -> Path:
        """
        Persist *df* to *dest_folder* using the filename copied from *copy_folder*
        that matches `file_id`.  Returns the full output path.
        """
        template = self._find_file(int(file_id), copy_folder)
        dest_path = dest_folder / template.name
        if return_path_only:
            return dest_path

        dest_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest_path, index=False)
        self.logger.info("Saved %s (%d rows) → %s", template.name, len(df), dest_path)
        return dest_path

    def get_all_file_ids_from_folder(self, folder: Path) -> List[int]:
        """Return every unique 5-digit file ID present in CSV filenames."""
        ids: set[int] = set()
        for file in folder.glob("*.csv"):
            if m := _ID_PATTERN.search(file.stem):
                ids.add(int(m.group(1)))
        return sorted(ids)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _load_initial_states(self) -> pd.DataFrame:
        """Load and concatenate initial-state CSVs from file or directory."""
        src = self._initial_state_src
        if isinstance(src, Path) and src.is_file():
            self.logger.debug("Loading initial states from %s", src)
            return pd.read_csv(src)

        dfs: list[pd.DataFrame] = []
        for csv_file in Path(src).glob("*.csv"):       # type: ignore[arg-type]
            dfs.append(pd.read_csv(csv_file))
        self.logger.debug("Loaded %d initial-state chunk(s)", len(dfs))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _find_file(self, file_id: int, folder: Path) -> Path:
        """Return the unique CSV path in *folder* whose stem includes the ID."""
        tag = f"-{file_id:05d}-"
        matches = [p for p in folder.glob("*.csv") if tag in p.stem]
        if not matches:
            raise FileNotFoundError(f"No CSV for ID {file_id} in {folder}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple CSVs for ID {file_id} in {folder}: {matches}")
        return matches[0]
