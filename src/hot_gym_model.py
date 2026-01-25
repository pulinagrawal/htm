import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.HTM import ColumnField, InputField
from brain import Brain
from encoder_layer.rdse import RDSEParameters
from encoder_layer.date_encoder import DateEncoderParameters    

config = {
    "num_columns": 256,
    "cells_per_column": 32,
    "resolution": .5,
    "rdse_seed": 5,
}

params = RDSEParameters(
    size=config["num_columns"],
    sparsity=0.02,
    resolution=config["resolution"],
    category=False,
    seed=config["rdse_seed"],
)
consumption_field = InputField(encoder_params=params)

date_params = DateEncoderParameters(
    day_of_week_radius=1,
    day_of_week_width=100,
    time_of_day_radius=1,
    time_of_day_width=100,
    weekend_width=20, 
)
date_field = InputField(encoder_params=date_params)

l1_column_field = ColumnField(
    input_fields=[consumption_field, date_field],
    non_spatial=True,
    num_columns=config["num_columns"],
    cells_per_column=config["cells_per_column"],
)

l2_column_field = ColumnField(
    input_fields=[l1_column_field],
    non_spatial=True,
    num_columns=config["num_columns"],
    cells_per_column=config["cells_per_column"],
)

l1_column_field.add_input_fields([l2_column_field])

# Create field manager for unified input handling
brain = Brain({
    "consumption": consumption_field,
    "date": date_field,
    "l1_column_field": l1_column_field,
    "l2_column_field": l2_column_field,
})

csv_path = Path(__file__).parent.parent / "data" / "rec-center-hourly.csv"
df = pd.read_csv(csv_path, skiprows=3, names=["timestamp", "kw_energy_consumption"])

def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string from CSV format to datetime."""
    return datetime.strptime(ts_str, "%m/%d/%y %H:%M")

df["datetime"] = df["timestamp"].apply(parse_timestamp)
burst_counts = []

for idx in tqdm(range(len(df)-500)):
    result = brain.step({
        "date": df["datetime"].iloc[idx],
        "consumption": df["kw_energy_consumption"].iloc[idx],
    })
    burst_counts.append(len(brain["l1_column_field"].bursting_columns))

print("Burst counts over time:", burst_counts)

brain.print_stats()

evaluation_bursts = []
errors = []
actual_values = []
predicted_values = []

for idx in tqdm(range(500)):
    index = (len(df)-500)+idx
    date = df["datetime"].iloc[index]
    value = df["kw_energy_consumption"].iloc[index]
    
    # Encode inputs first to get prediction before computing
    prediction = brain.prediction()["consumption"]
    brain.step({"date": date, "consumption": value})
    
    errors.append(abs(value - prediction)**2)
    actual_values.append(value)
    predicted_values.append(prediction)
    
    # Now compute (inputs already encoded)
    evaluation_bursts.append(len(brain["l1_column_field"].bursting_columns))


# mean abs error of predictions
mae = sum(errors[1:]) / len(errors[1:])
print("Error values:", errors)
print("Mean Absolute Error of predictions:", mae)
print("Evaluation bursting columns:", evaluation_bursts)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(actual_values, label='Actual', alpha=0.8)
plt.plot(predicted_values, label='Predicted', alpha=0.8)
plt.xlabel('Time Step')
plt.ylabel('kW Energy Consumption')
plt.title('HTM Predictions vs Actual Values')
plt.legend()
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=150)
plt.show()
