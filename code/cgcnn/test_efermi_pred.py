import pandas as pd

system = pd.read_csv("out_bl_atom05_s42/test_results_bl_atom05_s42_system.csv")
efermi = pd.read_csv("out_bl_atom05_s42/test_results_bl_atom05_s42_efermi.csv")

# Keep only valid targets.
def valid_mask(df):
    return df["mask"].astype(str).str.lower().isin(["true", "1"])

system = system.loc[
    valid_mask(system), ["mp-id", "target"]
].rename(columns={"target": "crystal_system"})

efermi = efermi.loc[
    valid_mask(efermi),
    ["mp-id", "target", "predicted_value"]
].rename(columns={"target": "efermi_true"})

# Match the two CSVs by Materials Project ID.
df = efermi.merge(
    system,
    on="mp-id",
    how="inner",
    validate="one_to_one",
)

# Compute absolute prediction errors.
df["abs_error"] = (
    df["efermi_true"] - df["predicted_value"]
).abs()

# Summarize by ground-truth crystal system.
summary = df.groupby("crystal_system").agg(
    count=("mp-id", "size"),
    mae=("abs_error", "mean"),
    mean=("efermi_true", "mean"),
    std=("efermi_true", "std"),
)

print(summary.round(4))

# Check against your reported overall MAE.
print("Matched overall MAE:", df["abs_error"].mean())
print("Matched samples:", len(df))
