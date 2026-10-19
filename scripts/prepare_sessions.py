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

# --------------------------------------------------
# LOAD SELECTED DEVICE DATA
# --------------------------------------------------

data_folder = Path('data')

df = pd.read_parquet(
    data_folder / 'selected_devices.parquet'
)

print('Dataset Shape:')
print(df.shape)

print('\nUnique Devices:')
print(df['device_id'].nunique())


# --------------------------------------------------
# INITIAL DATA QUALITY CHECKS
# --------------------------------------------------

print('\nTimestamp Statistics:')
print(df['timestamp'].describe())

print('\nBattery Level Statistics:')
print(df['battery_level'].describe())

print('\nBattery States:')
print(df['battery_state'].value_counts(dropna=False))

print('\nMissing Values:')
missing = df.isna().sum()
print(
    missing[
        missing > 0
    ].sort_values(ascending=False)
)


# --------------------------------------------------
# TIMESTAMP DISTRIBUTION
# --------------------------------------------------

print('\nObservations by Year:')
print(
    df['timestamp']
    .dt.year
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# BATTERY LEVEL VALIDITY
# --------------------------------------------------

print('\nBattery Level Validity:')
print('Battery < 0:',
      (df['battery_level'] < 0).sum())

print('Battery = 0:',
      (df['battery_level'] == 0).sum())

print('Battery between 1 and 100:',
      df['battery_level'].between(1, 100).sum())

print('Battery > 100:',
      (df['battery_level'] > 100).sum())


# --------------------------------------------------
# CPU USAGE VALIDITY
# --------------------------------------------------

print('\nCPU Usage Validity:')
print('Usage < 0:',
      (df['usage'] < 0).sum())

print('Usage between 0 and 1:',
      df['usage'].between(0, 1).sum())

print('Usage > 1:',
      (df['usage'] > 1).sum())


# --------------------------------------------------
# TEMPERATURE DISTRIBUTION
# --------------------------------------------------

print('\nTemperature Statistics:')
print(
    df['temperature'].describe(
        percentiles=[
            0.001,
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
            0.999
        ]
    )
)


# --------------------------------------------------
# INVESTIGATE TIMESTAMP ANOMALIES
# --------------------------------------------------

year_counts = df['timestamp'].dt.year.value_counts().sort_index()

main_period = df[
    df['timestamp'].dt.year.between(2017, 2020)
]

outside_period = df[
    ~df['timestamp'].dt.year.between(2017, 2020)
]

print('\nMain 2017-2020 Period:')
print('Rows:', len(main_period))
print('Percentage:',
      round(len(main_period) / len(df) * 100, 4))
print('Unique devices:',
      main_period['device_id'].nunique())

print('\nOutside 2017-2020:')
print('Rows:', len(outside_period))
print('Percentage:',
      round(len(outside_period) / len(df) * 100, 4))
print('Unique devices:',
      outside_period['device_id'].nunique())


print('\nOutside-period observations by year:')
print(
    outside_period['timestamp']
    .dt.year
    .value_counts()
    .sort_index()
)


print('\nDevices with most outside-period observations:')
print(
    outside_period['device_id']
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# INVALID BATTERY RECORDS
# --------------------------------------------------

print('\nBattery Levels Above 100:')
print(
    df.loc[
        df['battery_level'] > 100,
        [
            'device_id',
            'timestamp',
            'battery_state',
            'battery_level'
        ]
    ]
)


# --------------------------------------------------
# INVESTIGATE TEMPERATURE EXTREMES
# --------------------------------------------------

print('\nTemperature < 0:')
print((df['temperature'] < 0).sum())

print('Temperature < 5:')
print((df['temperature'] < 5).sum())

print('Temperature > 50:')
print((df['temperature'] > 50).sum())

print('Temperature > 60:')
print((df['temperature'] > 60).sum())

print('Temperature > 100:')
print((df['temperature'] > 100).sum())


print('\nTemperature Extremes by Year:')

temperature_extremes = df[
    (df['temperature'] < 0)
    | (df['temperature'] > 60)
]

print(
    temperature_extremes['timestamp']
    .dt.year
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# FINAL PREPROCESSING
# --------------------------------------------------

# Keep only valid battery percentages
df = df[
    df['battery_level'].between(1, 100)
].copy()

# CPU usage outside the expected 0-1 range is treated
# as invalid rather than deleting the entire observation
df['cpu_usage_clean'] = (
    df['usage']
    .where(df['usage'].between(0, 1))
)

df['cpu_usage_valid'] = (
    df['usage'].between(0, 1)
).astype(int)

print('\nAfter battery cleaning:')
print('Rows:', len(df))
print('Unique devices:', df['device_id'].nunique())


df = df.sort_values(
    ['device_id', 'timestamp']
).reset_index(drop=True)


# --------------------------------------------------
# CREATE LAGGED VARIABLES
# --------------------------------------------------

df['previous_device'] = df['device_id'].shift(1)
df['previous_battery_state'] = df['battery_state'].shift(1)
df['previous_timestamp'] = df['timestamp'].shift(1)
df['previous_battery_level'] = df['battery_level'].shift(1)

df['time_gap_minutes'] = (
    (df['timestamp'] - df['previous_timestamp'])
    .dt.total_seconds() / 60
)

df['battery_change'] = (
    df['battery_level']
    - df['previous_battery_level']
)


# --------------------------------------------------
# CREATE DISCHARGE SESSIONS
# --------------------------------------------------

# Identify discharging observations
is_discharging = df['battery_state'] == 'Discharging'

# Conditions that indicate the start of a new session
new_session = (
    is_discharging
    & (
        (df['device_id'] != df['previous_device'])
        | (df['previous_battery_state'] != 'Discharging')
        | (df['time_gap_minutes'] <= 0)
        | (df['time_gap_minutes'] > 60)
        | (df['battery_change'] > 5)
    )
)

# Give each new discharge session a unique number
df['session_id'] = new_session.cumsum()

# Keep only discharging observations
discharging = df[is_discharging].copy()


# --------------------------------------------------
# ENGINEER SESSION FEATURES
# --------------------------------------------------

discharging['manual_brightness'] = (
    discharging['screen_brightness']
    .where(discharging['screen_brightness'] >= 0)
)

discharging['auto_brightness'] = (
    discharging['screen_brightness'] == -1
).astype(int)

network_clean = (
    discharging['network_type']
    .str.upper()
)

discharging['wifi_network'] = (
    network_clean == 'WIFI'
).astype(int)

discharging['mobile_network'] = (
    network_clean
    .str.startswith('MOBILE', na=False)
    .astype(int)
)


sessions = (
    discharging
    .groupby(['device_id', 'session_id'])
    .agg(
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last'),

        start_battery=('battery_level', 'first'),
        end_battery=('battery_level', 'last'),

        observations=('battery_level', 'size'),

        avg_cpu_usage=('cpu_usage_clean', 'mean'),
        cpu_valid_proportion=('cpu_usage_valid', 'mean'),
        avg_temperature=('temperature', 'mean'),

        screen_on_proportion=('screen_on', 'mean'),
        avg_manual_brightness=('manual_brightness', 'mean'),
        auto_brightness_proportion=('auto_brightness', 'mean'),

        wifi_proportion=('wifi_network', 'mean'),
        mobile_proportion=('mobile_network', 'mean'),

        power_saver_proportion=('power_saver_enabled', 'mean')
    )
    .reset_index()
)


# --------------------------------------------------
# CALCULATE TARGET
# --------------------------------------------------

sessions['duration_hours'] = (
    (sessions['end_time'] - sessions['start_time'])
    .dt.total_seconds() / 3600
)

sessions['battery_drop'] = (
    sessions['start_battery']
    - sessions['end_battery']
)

sessions['discharge_rate'] = (
    sessions['battery_drop']
    / sessions['duration_hours']
)


# --------------------------------------------------
# APPLY SESSION QUALITY CRITERIA
# --------------------------------------------------

clean_sessions = sessions[
    (sessions['duration_hours'] >= 1.0)
    & (sessions['battery_drop'] >= 5)
    & (sessions['observations'] >= 3)
].copy()


# --------------------------------------------------
# FINAL SANITY CHECKS
# --------------------------------------------------

print('\nRaw Sessions:')
print(len(sessions))

print('\nClean Sessions:')
print(len(clean_sessions))

print('\nUnique Devices in Clean Sessions:')
print(clean_sessions['device_id'].nunique())

print('\nDischarge Rate Statistics:')
print(
    clean_sessions['discharge_rate'].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
)

print('\nMissing Values:')
missing = clean_sessions.isna().sum()
print(
    missing[
        missing > 0
    ].sort_values(ascending=False)
)

print('\nCPU Valid Proportion:')
print(
    clean_sessions['cpu_valid_proportion'].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
)

print('\nSessions with less than 90% valid CPU readings:')
print(
    (clean_sessions['cpu_valid_proportion'] < 0.90).sum()
)


# --------------------------------------------------
# SAVE FINAL SESSION DATASET
# --------------------------------------------------

output_file = data_folder / 'clean_sessions.parquet'

clean_sessions.to_parquet(
    output_file,
    index=False
)

print('\nSaved clean session dataset to:')
print(output_file)


# --------------------------------------------------
# INSPECT EXTREME DISCHARGE-RATE SESSIONS
# --------------------------------------------------

print('\nTop 20 Highest Discharge Rates:')

print(
    clean_sessions
    .nlargest(20, 'discharge_rate')
    [
        [
            'device_id',
            'start_time',
            'end_time',
            'observations',
            'start_battery',
            'end_battery',
            'duration_hours',
            'battery_drop',
            'discharge_rate'
        ]
    ]
    .to_string(index=False)
)


print('\nLowest 20 Discharge Rates:')

print(
    clean_sessions
    .nsmallest(20, 'discharge_rate')
    [
        [
            'device_id',
            'start_time',
            'end_time',
            'observations',
            'start_battery',
            'end_battery',
            'duration_hours',
            'battery_drop',
            'discharge_rate'
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# SESSION TELEMETRY COVERAGE
# --------------------------------------------------

print('\nSessions with no valid CPU data:')
print(clean_sessions['avg_cpu_usage'].isna().sum())

print('\nSessions with no temperature data:')
print(clean_sessions['avg_temperature'].isna().sum())

print('\nSessions with no power-saver data:')
print(clean_sessions['power_saver_proportion'].isna().sum())

print('\nSessions missing all three:')
missing_all_three = (
    clean_sessions['avg_cpu_usage'].isna()
    & clean_sessions['avg_temperature'].isna()
    & clean_sessions['power_saver_proportion'].isna()
)

print(missing_all_three.sum())


print('\nDischarge Rate - Sessions with telemetry:')
print(
    clean_sessions.loc[
        ~missing_all_three,
        'discharge_rate'
    ].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
)


print('\nDischarge Rate - Sessions missing telemetry:')
print(
    clean_sessions.loc[
        missing_all_three,
        'discharge_rate'
    ].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
)


# --------------------------------------------------
# CREATE FINAL MODELLING DATASET
# --------------------------------------------------

model_sessions = clean_sessions[
    clean_sessions['avg_cpu_usage'].notna()
    & clean_sessions['avg_temperature'].notna()
    & clean_sessions['power_saver_proportion'].notna()
].copy()

print('\nFinal Modelling Dataset:')
print('Sessions:', len(model_sessions))
print('Unique devices:', model_sessions['device_id'].nunique())

print('\nMissing Values in Final Modelling Dataset:')
missing = model_sessions.isna().sum()
print(
    missing[
        missing > 0
    ].sort_values(ascending=False)
)

print('\nFinal Target Statistics:')
print(
    model_sessions['discharge_rate'].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
)


# Save final modelling dataset
output_file = data_folder / 'model_sessions.parquet'

model_sessions.to_parquet(
    output_file,
    index=False
)

print('\nSaved final modelling dataset to:')
print(output_file)