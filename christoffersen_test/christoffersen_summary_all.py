import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# === Setup ===
output_dir = "christoffersen_test"
os.makedirs(output_dir, exist_ok=True)

# === Load Individual Summary CSVs ===
summary_files = {
    "GARCH": os.path.join("christoffersen_test", "garch_hit_summary.csv"),
    "LSTM-GARCH-VIX": os.path.join(
        "christoffersen_test", "lstm_garch_vix_hit_summary.csv"
    ),
    "LSTM-GARCH-EWMA-VIX": os.path.join(
        "christoffersen_test", "lstm_garch_ewma_vix_hit_summary.csv"
    ),
    "Pure LSTM": os.path.join("christoffersen_test", "pure_lstm_hit_summary.csv"),
}

# Read and combine summaries
summaries = []
for model, file_path in summary_files.items():
    df = pd.read_csv(file_path)
    if "Model" not in df.columns:
        df.insert(0, "Model", model)
    summaries.append(df)

summary_all = pd.concat(summaries, ignore_index=True)

# === Export Combined Summary ===
summary_csv = os.path.join(output_dir, "all_models_hit_summary.csv")
summary_txt = os.path.join(output_dir, "all_models_hit_summary.txt")
summary_all.to_csv(summary_csv, index=False)

with open(summary_txt, "w") as f:
    f.write(
        "| Model | Total Observations | Total Runs | Avg Run Length | Max Run Length | Min Run Length | Ljung-Box p-value (lag=10) | Plot Saved |\n"
    )
    f.write(
        "|-------|--------------------|------------|----------------|----------------|----------------|-----------------------------|-------------|\n"
    )
    for _, row in summary_all.iterrows():
        f.write(
            f"| {row['Model']} | {row['Total Observations']} | {row['Total Runs']} "
            f"| {row['Avg Run Length']:.2f} | {row['Max Run Length']} | {row['Min Run Length']} "
            f"| {row['Ljung-Box p-value (lag=10)']} | {row['Plot Saved']} |\n"
        )

# === Create 2x2 Plot Grid ===
plot_paths = {row["Model"]: row["Plot Saved"] for _, row in summary_all.iterrows()}
fig, axs = plt.subplots(4, 1, figsize=(16, 8))
axs = axs.flatten()

for i, (model, path) in enumerate(plot_paths.items()):
    if os.path.exists(path):
        img = mpimg.imread(path)
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(model)
    else:
        axs[i].text(0.5, 0.5, f"Missing plot: {model}", ha="center")
        axs[i].axis("off")

plt.tight_layout()
plot_grid_path = os.path.join(output_dir, "all_hit_sequence_plots.png")
plt.savefig(plot_grid_path, dpi=300)
plt.close()

print(f"✅ Combined summary saved to:\n- {summary_csv}\n- {summary_txt}")
print(f"🖼️ Combined plot saved to: {plot_grid_path}")
