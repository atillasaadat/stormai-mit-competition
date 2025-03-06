"""
datahandler.py

This module contains the DataHandler class, which is responsible for handling file I/O operations
related to initial state, OMNI, and density data. The class provides methods to read, process, and
save data from various folders.

Classes:
    DataHandler: Handles file I/O for initial state, OMNI, and density data.

Usage Example:
    import logging
    from pathlib import Path
    from datahandler import DataHandler

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Initialize DataHandler
    dh = DataHandler(
        logger,
        omni2_folder=Path("./data/omni2"),
        initial_state_folder=Path("./data/initial_state"),
        sat_density_folder=Path("./data/sat_density"),
        forcasted_omni2_folder=Path("./data/forcasted_omni2"),
        sat_density_omni_forcasted_folder=Path("./data/sat_density_omni_forcasted"),
        sat_density_omni_propagated_folder=Path("./data/sat_density_omni_propagated"),
    )

    # Example usage of DataHandler methods
    initial_state = dh.get_initial_state(file_id=1)
    omni_data = dh.read_csv_data(file_id=1, folder=dh.omni2_folder)
    dh.save_df_from_copy_folder_path(file_id=1, df=omni_data, copy_folder=dh.forcasted_omni2_folder, dest_folder=dh.sat_density_omni_forcasted_folder)
    file_ids = dh.get_all_file_ids_from_folder(folder=dh.sat_density_folder)
"""

import pandas as pd
from pathlib import Path
import logging

# =============================================================================
# DataHandler Class
# =============================================================================
class DataHandler:
    """
    Handles file I/O for initial state, OMNI, and density data.
    """

    def __init__(
        self, logger,
        omni2_folder,
        initial_state_folder,
        sat_density_folder,
        forcasted_omni2_folder,
        sat_density_omni_forcasted_folder,
        sat_density_omni_propagated_folder,
    ):
        """
        Initializes the DataHandler with the specified folders and logger.

        Args:
            logger (logging.Logger): Logger for logging messages.
            omni2_folder (Path): Path to the OMNI2 data folder.
            initial_state_folder (Path): Path to the initial state data folder.
            sat_density_folder (Path): Path to the satellite density data folder.
            forcasted_omni2_folder (Path): Path to the forecasted OMNI2 data folder.
            sat_density_omni_forcasted_folder (Path): Path to the forecasted satellite density OMNI data folder.
            sat_density_omni_propagated_folder (Path): Path to the propagated satellite density OMNI data folder.
        """
        self.logger = logger
        self.omni2_folder = omni2_folder
        self.initial_state_folder = initial_state_folder
        self.sat_density_folder = sat_density_folder
        self.forcasted_omni2_folder = forcasted_omni2_folder
        self.sat_density_omni_forcasted_folder = sat_density_omni_forcasted_folder
        self.sat_density_omni_propagated_folder = sat_density_omni_propagated_folder
        self.__read_initial_states()

    def __read_initial_states(self) -> None:
        """
        Reads initial state data from CSV files in the initial_state_folder and concatenates them into a single DataFrame.
        """
        dataframes = []
        for file in self.initial_state_folder.iterdir():
            if file.suffix == ".csv":
                df = pd.read_csv(file)
                dataframes.append(df)
        self.initial_states = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Loaded initial state data with {len(self.initial_states)} rows.")

    def get_initial_state(self, file_id):
        """
        Retrieves the initial state data for the specified file ID.

        Args:
            file_id (int): The file ID to retrieve the initial state for.

        Returns:
            pd.Series: The initial state data for the specified file ID.
        """
        return self.initial_states[self.initial_states["File ID"] == file_id].iloc[0]

    def read_csv_data(self, file_id: int, folder: Path) -> pd.DataFrame:
        """
        Reads CSV data for the specified file ID from the specified folder.

        Args:
            file_id (int): The file ID to read the data for.
            folder (Path): The folder to read the data from.

        Returns:
            pd.DataFrame: The data read from the CSV file.

        Raises:
            FileNotFoundError: If the CSV file for the specified file ID is not found.
        """
        file_id_str = f"{file_id:05d}"
        for file in folder.iterdir():
            if file.suffix == ".csv" and file_id_str in file.stem:
                self.logger.debug(f"Reading {folder.stem} data for File ID {file_id_str}.")
                data = pd.read_csv(file)
                data = data.sort_values(by="Timestamp").reset_index(drop=True)
                data["Timestamp"] = pd.to_datetime(data["Timestamp"])
                return data
        raise FileNotFoundError(f"{folder.stem} data for File ID {file_id} not found.")

    def save_df_from_copy_folder_path(self, file_id: int, df: pd.DataFrame, copy_folder: Path, dest_folder: Path) -> None:
        """
        Saves a DataFrame to a destination folder, using the filename from a copy folder.

        Args:
            file_id (int): The file ID to save the data for.
            df (pd.DataFrame): The DataFrame to save.
            copy_folder (Path): The folder to copy the filename from.
            dest_folder (Path): The destination folder to save the file to.

        Raises:
            FileNotFoundError: If the forecast file for the specified file ID is not found.
        """
        file_id_str = f"{file_id:05d}"
        output_file = None
        for file in copy_folder.iterdir():
            if file.suffix == ".csv" and file_id_str in file.stem:
                output_file = file.name
                break
        if output_file is None:
            raise FileNotFoundError(f"Could not locate forecast file for File ID {file_id}")

        output_path = dest_folder / output_file
        df.to_csv(output_path, index=False)
        self.logger.info(f"Result saved to {output_path} for file ID {file_id}.")

    def get_all_file_ids_from_folder(self, folder: Path) -> list:
        """
        Retrieves all file IDs from the folder.

        Args:
            folder (Path): The folder to retrieve file IDs from.

        Returns:
            list: List of file IDs.
        """
        file_ids = []
        for file in folder.iterdir():
            if file.suffix == ".csv" and file.stem.startswith("sat_density_"):
                file_id_str = file.stem.split("_")[-1]
                try:
                    file_ids.append(int(file_id_str))
                except ValueError:
                    self.logger.warning(f"Invalid file ID in filename: {file.name}")
        return file_ids

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    dh = DataHandler(logger,
        omni2_folder = Path("./data/omni2"),
        initial_state_folder = Path("./data/initial_state"),
        sat_density_folder = Path("./data/sat_density"),
        forcasted_omni2_folder = Path("./data/forcasted_omni2"),
        sat_density_omni_forcasted_folder = Path("./data/sat_density_omni_forcasted"),
        sat_density_omni_propagated_folder = Path("./data/sat_density_omni_propagated"),
    )
    from IPython import embed; embed(); quit()