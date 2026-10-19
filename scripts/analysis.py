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

df = pd.read_parquet('data/part-0001.parquet')

print('Dataset Shape:')
print(df.shape)

print('\nColumn Names:')
print(df.columns.tolist())

print('\nData Types:')
print(df.dtypes)

print('\nFirst 5 Rows:')
print(df.head(5))


# --------------------
# SANITY CHECKS
# --------------------

print('\nMissing Values:')
missing = df.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))

print('\nBattery Level Statistics:')
print(df['battery_level'].describe())

print('\nBattery States:')
print(df['battery_state'].value_counts(dropna=False))

print('\nTimestamp Statistics:')
print(df['timestamp'].describe())

print('\nUnique Devices:')
print(df['device_id'].nunique())

print('\nNetwork Type Values:')
print(df['network_type'].value_counts(dropna=False).head(20))

print('\nScreen Brightness Statistics:')
print(df['screen_brightness'].describe())

print('\nWiFi Signal Strength Statistics:')
print(df['wifi_signal_strength'].describe())

# --------------------
# WEIRD VALUES
# --------------------

print('\nScreen Brightness Value Counts:')
print(df['screen_brightness'].value_counts().sort_index().head(20))

print('\nScreen Brightness Special Values:')
print('Brightness = -1:',
      (df['screen_brightness'] == -1).sum())

print('\nWiFi Signal Strength Most Common Values:')
print(df['wifi_signal_strength'].value_counts().head(20))

print('\nWiFi Signal Special Values:')
print('Signal = -127:',
      (df['wifi_signal_strength'] == -127).sum())

print('Signal = -9999:',
      (df['wifi_signal_strength'] == -9999).sum())

print('\nWiFi Status Values:')
print(df['wifi_status'].value_counts(dropna=False))

print('\nWiFi Link Speed Statistics:')
print(df['wifi_link_speed'].describe())

print('\nWiFi Link Speed Most Common Values:')
print(df['wifi_link_speed'].value_counts().head(20))

# --------------------
# Constructing Discharging Sessions
# --------------------

# Sort observations by device and time
df = df.sort_values(
    ['device_id', 'timestamp']
).reset_index(drop=True)

# Previous observation information
df['previous_device'] = df['device_id'].shift(1)
df['previous_timestamp'] = df['timestamp'].shift(1)

# Calculate time since previous observation
df['time_gap_minutes'] = (
    (df['timestamp'] - df['previous_timestamp'])
    .dt.total_seconds() / 60
)

# Only keep gaps between observations belonging to the same device
same_device = df['device_id'] == df['previous_device']

device_gaps = df.loc[same_device, 'time_gap_minutes']

print('\nTime Gap Statistics Within Devices:')
print(device_gaps.describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
))

print('\nNon-positive Time Gaps:')
print((device_gaps <= 0).sum())

print('\nGaps > 30 minutes:')
print((device_gaps > 30).sum())

print('\nGaps > 60 minutes:')
print((device_gaps > 60).sum())

print('\nGaps > 120 minutes:')
print((device_gaps > 120).sum())

print('\nGaps > 360 minutes:')
print((device_gaps > 360).sum())

print('\nZero Time Gaps:')
print((device_gaps == 0).sum())

print('\nNegative Time Gaps:')
print((device_gaps < 0).sum())

# Previous battery state and level
df['previous_battery_state'] = df['battery_state'].shift(1)
df['previous_battery_level'] = df['battery_level'].shift(1)

# Consecutive observations must belong to the same device
same_device = df['device_id'] == df['previous_device']

# Both current and previous observations must be discharging
consecutive_discharging = (
    same_device
    & (df['battery_state'] == 'Discharging')
    & (df['previous_battery_state'] == 'Discharging')
)

# Battery change from previous observation
df['battery_change'] = (
    df['battery_level'] - df['previous_battery_level']
)

discharge_changes = df.loc[
    consecutive_discharging,
    'battery_change'
]

print('\nBattery Changes Between Consecutive Discharging Observations:')
print(discharge_changes.describe())

print('\nBattery Change Counts:')
print('Decreased:', (discharge_changes < 0).sum())
print('Unchanged:', (discharge_changes == 0).sum())
print('Increased:', (discharge_changes > 0).sum())

print('\nLarge Battery Increases (>5 percentage points):')
print((discharge_changes > 5).sum())

print('\nLarge Battery Decreases (<-5 percentage points):')
print((discharge_changes < -5).sum())

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

print('\nNumber of Raw Discharge Sessions:')
print(discharging['session_id'].nunique())

# Create variables needed for aggregation

# Clean CPU usage
# Valid CPU utilisation is represented as a proportion from 0 to 1.
# Values outside this range are treated as invalid measurements.
discharging['cpu_usage_clean'] = (
    discharging['usage']
    .where(discharging['usage'].between(0, 1))
)

# Record proportion of valid CPU readings within each session
discharging['cpu_usage_valid'] = (
    discharging['usage'].between(0, 1)
).astype(int)

# Manual brightness only
# -1 means automatic brightness, so exclude it from the manual average
discharging['manual_brightness'] = (
    discharging['screen_brightness']
    .where(discharging['screen_brightness'] >= 0)
)

# Indicator for automatic brightness
discharging['auto_brightness'] = (
    discharging['screen_brightness'] == -1
).astype(int)

# Network indicators
network_clean = (
    discharging['network_type']
    .str.upper()
)

discharging['wifi_network'] = (
    network_clean == 'WIFI'
).astype(int)

discharging['mobile_network'] = (
    network_clean == 'MOBILE'
).astype(int)


# Aggregate observations into discharge sessions
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

# Session duration in hours
sessions['duration_hours'] = (
    (sessions['end_time'] - sessions['start_time'])
    .dt.total_seconds() / 3600
)

# Total battery percentage-point decrease
sessions['battery_drop'] = (
    sessions['start_battery']
    - sessions['end_battery']
)

# Target variable: average battery discharge rate
sessions['discharge_rate'] = (
    sessions['battery_drop']
    / sessions['duration_hours']
)

print('\nSession Dataset Shape:')
print(sessions.shape)

print('\nSession Duration Statistics:')
print(sessions['duration_hours'].describe())

print('\nBattery Drop Statistics:')
print(sessions['battery_drop'].describe())

print('\nDischarge Rate Statistics:')
print(sessions['discharge_rate'].describe())

print('\nObservation Count Per Session:')
print(sessions['observations'].describe())

print('\nFirst 10 Sessions:')
print(
    sessions[
        [
            'device_id',
            'session_id',
            'start_time',
            'end_time',
            'observations',
            'start_battery',
            'end_battery',
            'duration_hours',
            'battery_drop',
            'discharge_rate'
        ]
    ].head(10)
)

# --------------------------------------------------
# EVALUATE SESSION QUALITY THRESHOLDS
# --------------------------------------------------

thresholds = [
    (0.25, 2),   # 15 minutes, 2 percentage points
    (0.50, 3),   # 30 minutes, 3 percentage points
    (0.50, 5),   # 30 minutes, 5 percentage points
    (1.00, 5),   # 60 minutes, 5 percentage points
    (1.00, 10)   # 60 minutes, 10 percentage points
]

print('\nSession Quality Threshold Comparison:')

for min_duration, min_drop in thresholds:

    filtered = sessions[
        (sessions['duration_hours'] >= min_duration)
        & (sessions['battery_drop'] >= min_drop)
        & (sessions['observations'] >= 3)
    ]

    print(
        f'\nMinimum duration: {min_duration} hours'
        f' | Minimum battery drop: {min_drop}%'
    )

    print('Sessions retained:', len(filtered))

    if len(filtered) > 0:
        print(
            'Median discharge rate:',
            round(filtered['discharge_rate'].median(), 2)
        )

        print(
            '95th percentile discharge rate:',
            round(filtered['discharge_rate'].quantile(0.95), 2)
        )

        print(
            'Maximum discharge rate:',
            round(filtered['discharge_rate'].max(), 2)
        )


# --------------------------------------------------
# FINAL SESSION QUALITY FILTER
# --------------------------------------------------

clean_sessions = sessions[
    (sessions['duration_hours'] >= 1.0)
    & (sessions['battery_drop'] >= 5)
    & (sessions['observations'] >= 3)
].copy()

print('\nClean Session Dataset Shape:')
print(clean_sessions.shape)

print('\nUnique Devices in Clean Dataset:')
print(clean_sessions['device_id'].nunique())

print('\nClean Discharge Rate Statistics:')
print(clean_sessions['discharge_rate'].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
))

print('\nMissing Values in Clean Session Dataset:')
missing_sessions = clean_sessions.isna().sum()
print(
    missing_sessions[
        missing_sessions > 0
    ].sort_values(ascending=False)
)

print('\nPredictor Statistics:')
predictors = [
    'avg_cpu_usage',
    'avg_temperature',
    'screen_on_proportion',
    'avg_manual_brightness',
    'auto_brightness_proportion',
    'wifi_proportion',
    'mobile_proportion',
    'power_saver_proportion'
]

print(clean_sessions[predictors].describe().T)

# --------------------------------------------------
# INVESTIGATE CPU USAGE VARIABLE
# --------------------------------------------------

print('\nRaw CPU Usage Statistics:')
print(df['usage'].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
))

print('\nMost Common CPU Usage Values:')
print(df['usage'].value_counts().head(20))

print('\nCPU Usage Value Ranges:')
print('Usage < 0:', (df['usage'] < 0).sum())
print('Usage = -1:', (df['usage'] == -1).sum())
print('Usage between 0 and 1:',
      df['usage'].between(0, 1, inclusive='both').sum())
print('Usage > 1:', (df['usage'] > 1).sum())
print('Usage > 10:', (df['usage'] > 10).sum())
print('Usage > 100:', (df['usage'] > 100).sum())

print('\nCPU Usage for Discharging Observations:')
print(
    discharging['usage'].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
)

print('\nRaw Temperature Statistics:')
print(df['temperature'].describe(
    percentiles=[0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
))

print('\nTemperature Value Ranges:')
print('Temperature < 0:', (df['temperature'] < 0).sum())
print('Temperature < 5:', (df['temperature'] < 5).sum())
print('Temperature > 50:', (df['temperature'] > 50).sum())
print('Temperature > 100:', (df['temperature'] > 100).sum())


