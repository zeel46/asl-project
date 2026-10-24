import pandas as pd
import matplotlib.pyplot as plt

kaggle = pd.read_csv("asl_landmarks.csv")
kaggle["source"] = "Kaggle (1 person)"
hg = pd.read_csv("asl_hg_landmarks.csv", dtype={"label": str})
hg["source"] = "ASL-HG (10 people)"
df = pd.concat([kaggle, hg.drop(columns=["person"])], ignore_index=True)

counts = df.groupby(["label", "source"]).size().unstack(fill_value=0)
counts = counts.loc[sorted(counts.index, key=lambda s: (s.isalpha(), s))]
ax = counts.plot(kind="bar", stacked=True, figsize=(14, 5), color=["#7fb3e0", "#8fd3b6"])
ax.set_title(f"Samples per sign ({len(df)} hands in total)")
ax.set_xlabel("Sign")
ax.set_ylabel("Number of hands")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data_distribution.png", dpi=130)

print(f"{len(df)} hands, {counts.shape[0]} signs")
print("Saved data_distribution.png")
plt.show()
