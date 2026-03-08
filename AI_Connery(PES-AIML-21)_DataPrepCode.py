import pandas as pd
import numpy as np
from faker import Faker
from scipy import stats
from datetime import datetime   # used for unique excel file name

NUM_RECORDS = 1300              # total rows
ANOMALY_RATE = 0.08             # 8% anomalies
np.random.seed(42)
fake = Faker()

# CLEAN DATA GENERATION
def generate_clean_data(n):
    df = pd.DataFrame({
        "Patient_ID": [f"PID{1000+i}" for i in range(n)],      # unique patient id
        "Patient_Name": [fake.name() for _ in range(n)],       # synthetic patient names
        "Age": np.clip(np.random.normal(50, 15, n), 18, 90).astype(int),  # valid age range
        "Gender": np.random.choice(["Male", "Female"], n),     # valid gender values
        "Ethnicity": np.random.choice(["Asian", "White", "Black", "Hispanic"], n),
        "Region": np.random.choice(["Urban", "Rural"], n),
        "Height_m": np.round(np.random.normal(1.65, 0.1, n), 2),
        "Weight_kg": np.round(np.random.normal(70, 15, n), 2),
    })
    df["BMI"] = np.round(df["Weight_kg"] / (df["Height_m"] ** 2), 2)   # derived column
    df["Smoking"] = np.random.choice([0, 1, 2], n)
    df["Alcohol"] = np.random.choice([0, 1, 2], n)
    df["Physical_Activity"] = np.random.choice([0, 1, 2], n)
    df["Family_History"] = np.random.choice([0, 1], n)
    df["Radiation_Exposure"] = np.random.choice([0, 1], n)
    df["Chronic_Pain"] = np.random.choice([0, 1], n)
    df["Weight_Loss"] = np.random.choice([0, 1], n)
    df["Tumor_Size_cm"] = np.round(np.random.uniform(0.5, 5.0, n), 2)
    df["Histopathology"] = np.round(df["Tumor_Size_cm"] * 1.3, 2)      # dependent value
    df["Genetic_Marker"] = np.where(df["Tumor_Size_cm"] > 2.5, 1, 0)
    df["Imaging"] = np.where(df["Tumor_Size_cm"] > 2, 1, 0)
    df["Tumor_Marker"] = np.round(10 + df["Tumor_Size_cm"] * 6, 2)
    df["Nodule_Presence"] = np.where(df["Tumor_Size_cm"] > 1.5, 1, 0)
    df["Hemoglobin"] = np.round(15 - df["Tumor_Size_cm"] * 0.4, 2)
    df["WBC"] = np.round(6000 + df["Tumor_Marker"] * 18, 2)
    df["Cancer"] = np.where(
        (df["Tumor_Size_cm"] > 2) &
        (df["Genetic_Marker"] == 1) &
        (df["Imaging"] == 1),
        "YES", "NO"       # target label
    )
    return df

# ANOMALY INTRODUCTION 
def introduce_anomalies(df):
    df_dirty = df.copy()                     # duplicate original data
    n = int(ANOMALY_RATE * len(df_dirty))    # anomaly count
    # missing value anomalies
    for col in ["BMI", "Tumor_Size_cm", "WBC"]:
        df_dirty.loc[df_dirty.sample(n).index, col] = np.nan
    # unrealistic age values
    df_dirty.loc[df_dirty.sample(n).index, "Age"] = np.random.choice(
        [130, 150, 200, 999], n
    )
    # extreme tumor values
    df_dirty.loc[df_dirty.sample(n).index, "Tumor_Size_cm"] = np.random.choice(
        [10, 20, 50], n
    )
    # BMI outliers
    df_dirty.loc[df_dirty.sample(n).index, "BMI"] = np.random.choice(
        [55, 60, 100], n
    )
    # datatype error anomalies
    df_dirty.loc[df_dirty.sample(n).index, "Age"] = "unknown"
    df_dirty.loc[df_dirty.sample(n).index, "Tumor_Size_cm"] = "large"
    df_dirty.loc[df_dirty.sample(n).index, "WBC"] = "high"
    # invalid gender values
    df_dirty.loc[df_dirty.sample(n).index, "Gender"] = np.random.choice(
        ["Unknown", "M", "malee", "FEMALEE"], n
    )
    # wrong region spellings
    df_dirty.loc[df_dirty.sample(n).index, "Region"] = np.random.choice(
        ["Metro", "urbn", "rural "], n
    )
    # category typo
    df_dirty.loc[df_dirty.sample(n).index, "Ethnicity"] = "Asain"
    df_dirty.loc[df_dirty.sample(n).index, "Cancer"] = "YESS"
    # human typing mistakes
    df_dirty.loc[df_dirty.sample(n).index, "Gender"] = np.random.choice(
        ["Femlae", "MALE ", " femael", "Mle", "FeMale"], n
    )
    df_dirty.loc[df_dirty.sample(n).index, "Cancer"] = np.random.choice(
        ["Yse", "N o", "Yes ", "Noo", "yess"], n
    )
    # duplicate record anomaly
    df_dirty = pd.concat(
        [df_dirty, df_dirty.sample(n // 2)],
        ignore_index=True
    )
    return df_dirty

#  CLEANING PROCESS
def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)   # remove duplicates
    numeric_cols = [
        "Age", "BMI", "Height_m", "Weight_kg",
        "Tumor_Size_cm", "Hemoglobin", "WBC"
    ]
    # convert datatype errors to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # standardize gender values
    df["Gender"] = df["Gender"].replace({
        "M": "Male", "malee": "Male", "MALE ": "Male", "Mle": "Male",
        "FEMALEE": "Female", "Femlae": "Female", " femael": "Female",
        "FeMale": "Female", "Unknown": np.nan
    })
    # region correction
    df["Region"] = df["Region"].replace({
        "urbn": "Urban", "rural ": "Rural", "Metro": np.nan
    })
    # spelling correction
    df["Ethnicity"] = df["Ethnicity"].replace({"Asain": "Asian"})
    # normalize cancer labels
    df["Cancer"] = df["Cancer"].replace({
        "YESS": "YES", "Yse": "YES", "Yes ": "YES",
        "yess": "YES", "N o": "NO", "Noo": "NO"
    })
    # rule-based cleaning
    df.loc[(df["Age"] < 18) | (df["Age"] > 120), "Age"] = np.nan
    df.loc[df["Height_m"] <= 0, "Height_m"] = np.nan
    df.loc[df["Tumor_Size_cm"] < 0, "Tumor_Size_cm"] = np.nan
    # median imputation for numeric values
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    # mode imputation for categorical values
    for col in ["Gender", "Region", "Ethnicity", "Cancer"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    # recompute BMI after cleaning
    df["BMI"] = np.round(df["Weight_kg"] / (df["Height_m"] ** 2), 2)
    # IQR outlier removal
    for col in ["BMI", "Tumor_Size_cm"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df[col] >= Q1 - 1.5 * IQR) &
            (df[col] <= Q3 + 1.5 * IQR)
        ]
    # Z-score outlier removal
    df = df[np.abs(stats.zscore(df["WBC"])) < 3]

    return df.reset_index(drop=True)

#  VALIDATION CHECK
def validate_dataset(df):

    numeric_expected_float = [
        "BMI","Height_m","Weight_kg",
        "Tumor_Size_cm","Hemoglobin","WBC"
    ]

    numeric_expected_int = [
        "Age","Smoking","Alcohol","Physical_Activity",
        "Family_History","Radiation_Exposure",
        "Chronic_Pain","Weight_Loss",
        "Genetic_Marker","Imaging","Nodule_Presence"
    ]

    for col in numeric_expected_float:
        assert np.issubdtype(df[col].dtype, np.number), f"{col} contains non-numeric values"

    for col in numeric_expected_int:
        assert (df[col] % 1 == 0).all(), f"{col} contains non-integer values"

    assert df["Gender"].isin(["Male", "Female"]).all(), "invalid gender values"
    assert df["Age"].between(18, 120).all(), "invalid age values"
    assert df["Cancer"].isin(["YES", "NO"]).all(), "invalid cancer labels"
    print("Validation Passed - dataset is clean")

#MAIN EXECUTION 
def main():
    df_clean = generate_clean_data(NUM_RECORDS)     # original data
    df_dirty = introduce_anomalies(df_clean)        # dataset with anomalies
    df_final = clean_data(df_dirty)                 # cleaned dataset
    validate_dataset(df_final)                      # validation step
    filename = f"Cancer_Patient_Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"  # avoid overwrite
    with pd.ExcelWriter(filename) as writer:        # save all datasets
        df_clean.to_excel(writer, sheet_name="Original_Clean_Data", index=False)
        df_dirty.to_excel(writer, sheet_name="Data_With_Anomalies", index=False)
        df_final.to_excel(writer, sheet_name="Cleaned_Final_Data", index=False)
    print("Excel file created:", filename)

if __name__ == "__main__":
    main()
