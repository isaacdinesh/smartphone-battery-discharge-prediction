import sys
from pathlib import Path

class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)

    def flush(self):
        for output in self.outputs:
            output.flush()

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

log_file = open(
    results_folder / f"{Path(__file__).stem}_output.txt",
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(sys.stdout, log_file)


import pandas as pd
from pathlib import Path

# --------------------------------------------------
# LOCATE ALL SAMPLE PARQUET FILES
# --------------------------------------------------

data_folder = Path('data')

sample_files = sorted(
    data_folder.glob('part-*.parquet')
)

print('Number of sample files found:')
print(len(sample_files))


# --------------------------------------------------
# COLLECT UNIQUE DEVICE IDs
# --------------------------------------------------

all_device_ids = set()

for i, file in enumerate(sample_files, start=1):

    # Only read device_id instead of loading all 38 columns
    device_data = pd.read_parquet(
        file,
        columns=['device_id']
    )

    all_device_ids.update(
        device_data['device_id'].dropna().unique()
    )

    print(
        f'Processed file {i}/{len(sample_files)}'
        f' | Unique devices so far: {len(all_device_ids)}'
    )


print('\nTotal Unique Device IDs:')
print(len(all_device_ids))


# --------------------------------------------------
# RANDOMLY SAMPLE DEVICES
# --------------------------------------------------

import numpy as np

RANDOM_SEED = 42
N_DEVICES = 5000

rng = np.random.default_rng(RANDOM_SEED)

all_device_ids_array = np.array(
    sorted(all_device_ids)
)

selected_device_ids = rng.choice(
    all_device_ids_array,
    size=N_DEVICES,
    replace=False
)

selected_device_ids = set(selected_device_ids)

print('\nSelected Device IDs:')
print(len(selected_device_ids))


# --------------------------------------------------
# EXTRACT ALL RECORDS FOR SELECTED DEVICES
# --------------------------------------------------

selected_data = []

for i, file in enumerate(sample_files, start=1):

    part = pd.read_parquet(file)

    selected_part = part[
        part['device_id'].isin(selected_device_ids)
    ].copy()

    selected_data.append(selected_part)

    print(
        f'Extracted file {i}/{len(sample_files)}'
        f' | Rows retained: {len(selected_part)}'
    )


# Combine records from all 70 files
df = pd.concat(
    selected_data,
    ignore_index=True
)

# Free the list containing the individual DataFrames
del selected_data


print('\nCombined Dataset Shape:')
print(df.shape)

print('\nUnique Devices Retained:')
print(df['device_id'].nunique())

print('\nTimestamp Range:')
print(df['timestamp'].min())
print(df['timestamp'].max())


# --------------------------------------------------
# SORT COMPLETE DEVICE HISTORIES
# --------------------------------------------------

df = df.sort_values(
    ['device_id', 'timestamp']
).reset_index(drop=True)

print('\nDataset sorted by device and timestamp.')


# --------------------------------------------------
# SAVE DEVICE SAMPLE
# --------------------------------------------------

output_file = data_folder / 'selected_devices.parquet'

df.to_parquet(
    output_file,
    index=False
)

print('\nSaved selected-device dataset to:')
print(output_file)