import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tqdm import tqdm
from src.HTM import ColumnField, InputField
from src.brain import Brain
from visualizer import HTMVisualizer
from encoder_layer.rdse import RDSEParameters
from encoder_layer.date_encoder import DateEncoderParameters    

config = {
    "num_columns": 1024,
    "cells_per_column": 32,
    "radius": .5,
    "rdse_seed": 5,
}

params = RDSEParameters(
    size=config["num_columns"],
    sparsity=0.02,
    radius=config["radius"],
    category=False,
    seed=config["rdse_seed"],
)
consumption_field = InputField(encoder_params=params)

date_params = DateEncoderParameters(
    day_of_week_radius=1,
    day_of_week_size=100,
    time_of_day_radius=1,
    time_of_day_size=100,
    weekend_size=20, 
)
date_field = InputField(encoder_params=date_params)
column_field = ColumnField(
    input_fields=[consumption_field],
    non_spatial=True,
    num_columns=config["num_columns"],
    cells_per_column=config["cells_per_column"],
)

brain  = Brain({"consumption": consumption_field, "columns": column_field})

csv_path = Path(__file__).parent.parent / "data" / "rec-center-hourly.csv"
df = pd.read_csv(csv_path, skiprows=3, names=["timestamp", "kw_energy_consumption"])

def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string from CSV format to datetime."""
    return datetime.strptime(ts_str, "%m/%d/%y %H:%M")

df["datetime"] = df["timestamp"].apply(parse_timestamp)
burst_counts = []

def generate_input_sequence():
    for idx in tqdm(range(len(df)-500)):
        date = df["datetime"].iloc[idx]
        consumption = df["kw_energy_consumption"].iloc[idx]
        yield {"consumption": consumption}

column_field.print_stats()

evaluation_bursts = []
errors = []
actual_values = []
predicted_values = []

viz = HTMVisualizer(
    brain,
    input_sequence=generate_input_sequence(),
    title="Hot Gym Energy Consumption HTM Visualizer",
)
viz.run()
