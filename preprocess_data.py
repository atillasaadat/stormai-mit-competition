import numpy as np
import pandas as pd
from pathlib import Path
from IPython import embed

def read_initial_states(initial_state_folder):
    all_dataframes = []
    for file in initial_state_folder.iterdir():
        if file.suffix == '.csv':
            df = pd.read_csv(file)
            all_dataframes.append(df)
    combined_dataframe = pd.concat(all_dataframes, ignore_index=True)
    return combined_dataframe

if __name__ == "__main__":
    goes_folder = Path("./data/goes")
    omni_folder = Path("./data/omni")
    initial_state_folder = Path("./data/initial_state")
    sat_density_folder = Path("./data/sat_density")
    combined_data_folder = Path("./data/combined_data")
    forcasted_omni2_data_folder = Path("./data/forcasted_omni2")

    embed();quit()