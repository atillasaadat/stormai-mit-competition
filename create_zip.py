#!/usr/bin/env python3
import os
import zipfile
from pathlib import Path
from datetime import datetime

def zip_items(output_zip, items):
    """
    Create a zip archive (output_zip) containing copies of the given list of items.
    If an item is a directory, include all its contents recursively.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if os.path.exists(item):
                if os.path.isdir(item):
                    # Walk through directory and add each file with a path relative to the item folder.
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            full_path = os.path.join(root, file)
                            # Compute arcname relative to current working directory.
                            arcname = os.path.relpath(full_path, os.getcwd())
                            zf.write(full_path, arcname)
                elif os.path.isfile(item):
                    # Add file using its basename.
                    zf.write(item, os.path.basename(item))
            else:
                print(f"Warning: '{item}' does not exist and will be skipped.")
    print(f"Created zip archive: {output_zip}")


def main():
    # List of files and folders to include in the zip archive.
    # items_to_zip = [
    #     "pymsis/",
    #     "datahandler.py",
    #     "density_prediction_patchtst_model.pth",
    #     "environment.yml",
    #     "scaling_params.json",
    #     "setup.py",
    #     "submission.py",
    #     "SW-All.csv",
    #     "best_forecast_model.pth",
    #     "scaler.pkl",
    #     "best_params.json",
    #     "shri_propagator.py",
    #     "forecast_omni_v2.py"
    # ]
    items_to_zip = [
        #"karman/",
        "datahandler.py",
        "environment.yml",
        "setup.py",
        "submission.py",
        #"shri_propagator.py",
        #"density_models.py",
        #"nn.py",
        #"util.py",
        #"ts_karman_model_tft_ss65_heads4_lag10000_resolution100_valid_mape_15.059_params_1074865.torch",
        #"ts_data_normalized.pk",
        "density_net.pt"
    ]
    # Create a filename with the current date and time in file-friendly format.
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_zip = f"submission_{now}.zip"
    
    zip_items(output_zip, items_to_zip)

if __name__ == "__main__":
    main()
