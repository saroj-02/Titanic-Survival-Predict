import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# ============================================================
# 1. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("output", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_PATH = "data/Titanic-Dataset.csv"

print("=" * 60)
print("TITANIC SURVIVAL - DATA CLEANING PROJECT")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")


# ============================================================
# 3. DISPLAY BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print(df.describe())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print("\nMissing percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))


# ============================================================
# 6. HANDLE MISSING DATA
# ============================================================

print("\n" + "=" * 60)
print("HANDLING MISSING DATA")
print("=" * 60)


# ------------------------------------------------------------
# Age
# Use median because Age may contain outliers
# ------------------------------------------------------------

age_median = df["Age"].median()

df["Age"] = df["Age"].fillna(age_median)

print(f"\nMissing Age values filled using median: {age_median}")


# ------------------------------------------------------------
# Fare
# Use median
# ------------------------------------------------------------

fare_median = df["Fare"].median()

df["Fare"] = df["Fare"].fillna(fare_median)

print(f"Missing Fare values filled using median: {fare_median}")


# ------------------------------------------------------------
# Embarked
# Use mode because it is categorical
# ------------------------------------------------------------

embarked_mode = df["Embarked"].mode()[0]

df["Embarked"] = df["Embarked"].fillna(embarked_mode)

print(f"Missing Embarked values filled using mode: {embarked_mode}")


# ------------------------------------------------------------
# Cabin
# Cabin contains a large number of missing values.
# It is removed from this basic data-cleaning project.
# ------------------------------------------------------------

if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

    print("Cabin column removed because of a high percentage of missing values.")


# ============================================================
# 7. CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 8. ENCODE SEX USING LABEL ENCODER
# ============================================================

print("\n" + "=" * 60)
print("ENCODING SEX")
print("=" * 60)

label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])

print("\nSex encoding:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {index}")


# ============================================================
# 9. ENCODE EMBARKED USING ONE HOT ENCODER
# ============================================================

print("\n" + "=" * 60)
print("ENCODING EMBARKED")
print("=" * 60)

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

embarked_encoded = one_hot_encoder.fit_transform(
    df[["Embarked"]]
)

encoded_columns = one_hot_encoder.get_feature_names_out(
    ["Embarked"]
)

encoded_df = pd.DataFrame(
    embarked_encoded,
    columns=encoded_columns,
    index=df.index
)

# Add encoded columns
df = pd.concat([df, encoded_df], axis=1)

# Remove original Embarked column
df.drop("Embarked", axis=1, inplace=True)

print("\nCreated encoded columns:")
print(list(encoded_columns))


# ============================================================
# 10. VISUALIZE AGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CREATING AGE DISTRIBUTION VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

visualization_path = "visualizations/age_distribution.png"

plt.savefig(
    visualization_path,
    dpi=300
)

plt.close()

print(
    f"Age distribution saved to: {visualization_path}"
)


# ============================================================
# 11. OPTIONAL SURVIVAL VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.tight_layout()

survival_plot_path = "visualizations/survival_distribution.png"

plt.savefig(
    survival_plot_path,
    dpi=300
)

plt.close()

print(
    f"Survival distribution saved to: {survival_plot_path}"
)


# ============================================================
# 12. DISPLAY FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLEANED DATASET")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())


# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================

OUTPUT_PATH = "output/cleaned_titanic.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print(
    f"\nCleaned dataset saved successfully at:\n{OUTPUT_PATH}"
)

print(
    f"\nVisualization saved at:\n{visualization_path}"
)

print(
    f"\nSurvival visualization saved at:\n{survival_plot_path}"
)