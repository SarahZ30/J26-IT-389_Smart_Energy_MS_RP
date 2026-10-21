import pandas as pd
import glob
import os

files = sorted(glob.glob("CLEAN_House*.csv"))

summary_rows = []

for file in files:
    print(f"Processing: {file}")

    df = pd.read_csv(file)

    house_name = os.path.splitext(os.path.basename(file))[0]

    # Basic information
    row = {
        "House": house_name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Start_Time": df["Time"].iloc[0] if "Time" in df.columns else None,
        "End_Time": df["Time"].iloc[-1] if "Time" in df.columns else None,
        "Total_Missing_Values": int(df.isna().sum().sum()),
        "Issue_Rows": int((df["Issues"] == 1).sum()) if "Issues" in df.columns else 0,
    }

    # Aggregate statistics
    if "Aggregate" in df.columns:
        row["Aggregate_Min"] = df["Aggregate"].min()
        row["Aggregate_Max"] = df["Aggregate"].max()
        row["Aggregate_Mean"] = df["Aggregate"].mean()

    # Appliance activity statistics
    appliance_cols = [c for c in df.columns if c.startswith("Appliance")]

    for col in appliance_cols:
        row[f"{col}_NonZero_Count"] = int((df[col] > 0).sum())
        row[f"{col}_Mean_Watts"] = df[col].mean()
        row[f"{col}_Max_Watts"] = df[col].max()

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv("refit_house_summary.csv", index=False)

print("\nDone.")
print("Created: refit_house_summary.csv")