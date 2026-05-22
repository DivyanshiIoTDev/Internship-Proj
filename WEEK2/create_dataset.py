
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# NUMBER OF RECORDS
# -----------------------------
n = 5000

# -----------------------------
# TIMESTAMPS
# -----------------------------
start_time = datetime.now()

timestamps = [
    start_time + timedelta(minutes=i)
    for i in range(n)
]

# -----------------------------
# SENSOR DATA
# -----------------------------

temperature = np.random.normal(35, 5, n)

humidity = np.random.normal(60, 10, n)

vibration = np.random.normal(1.2, 0.4, n)

power = np.random.normal(6, 2, n)

# -----------------------------
# ADD MISSING VALUES
# -----------------------------

for col in [temperature, humidity, vibration, power]:

    for i in range(random.randint(40, 80)):
        col[random.randint(0, n-1)] = np.nan

# -----------------------------
# ADD OUTLIERS / ANOMALIES
# -----------------------------

for i in range(50):

    temperature[random.randint(0, n-1)] = random.uniform(100, 180)

    humidity[random.randint(0, n-1)] = random.uniform(-20, 150)

    vibration[random.randint(0, n-1)] = random.uniform(5, 15)

    power[random.randint(0, n-1)] = random.uniform(20, 80)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

df = pd.DataFrame({

    "Temperature_C": temperature,

    "Humidity_percent": humidity,

    "Vibration_mms": vibration,

    "Power_Consumption_kW": power

})

# -----------------------------
# ADD DUPLICATE ROWS
# -----------------------------

duplicates = df.sample(50)

df = pd.concat([df, duplicates])

# -----------------------------
# SHUFFLE DATASET
# -----------------------------

df = df.sample(frac=1).reset_index(drop=True)

# -----------------------------
# SAVE DATASET
# -----------------------------

df.to_csv("advanced_sensor_dataset.csv", index=False)

# -----------------------------
# DISPLAY INFORMATION
# -----------------------------

print("Dataset Generated Successfully!")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())