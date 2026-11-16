
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Find the training history file
files = list(
    Path("runs/testingConvNext/experiment-10epochs")
    .glob("*/history.csv")
)

# Load training history
df = pd.read_csv(files[0])

print("Available columns:")
print(df.columns.tolist())

# Select all numeric columns except the epoch number
epoch_col = "epoch"
metric_cols = [
    col for col in df.select_dtypes(include="number").columns
    if col != epoch_col
]

# Create a directory for generated plots
output_dir = Path("convnext_plots")
output_dir.mkdir(exist_ok=True)

# Generate and save a plot for each metric
for col in metric_cols:
    plt.figure(figsize=(8, 4))
    plt.plot(df[epoch_col], df[col], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(col)
    plt.title(f"ConvNeXt Tiny - {col}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{col}.png", dpi=150)
    plt.close()

print("Plots saved to:", output_dir.resolve())
