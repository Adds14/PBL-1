import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# 1. Load dataset
df = pd.read_csv("Data/bank.csv", sep=';')
data = df.copy()

# 2. Encode categorical columns (handle both 'object' and 'str' dtypes)
label_encoders = {}
cat_cols = data.select_dtypes(include=['object', 'str']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 3. Ask user for number of synthetic rows
n_samples = int(input("Enter number of synthetic rows to generate: "))

# 4. Calculate target split: 70% no, 30% yes
y_col = 'y'
y_encoder = label_encoders[y_col]

no_encoded  = y_encoder.transform(["no"])[0]
yes_encoded = y_encoder.transform(["yes"])[0]

n_no  = int(n_samples * 0.70)
n_yes = n_samples - n_no

# 5. Separate data by class
data_no  = data[data[y_col] == no_encoded].values
data_yes = data[data[y_col] == yes_encoded].values

# 6. KNN-based SMOTE generation per class
def generate_samples(class_data, n, k=5):
    knn = NearestNeighbors(n_neighbors=min(k, len(class_data)))
    knn.fit(class_data)
    samples = []
    for _ in range(n):
        idx = np.random.randint(0, len(class_data))
        x = class_data[idx].reshape(1, -1)
        neighbors = knn.kneighbors(x, return_distance=False)
        neighbor = class_data[np.random.choice(neighbors[0][1:])]
        gap = np.random.rand()
        synthetic = x.flatten() + gap * (neighbor - x.flatten())
        samples.append(synthetic)
    return samples

# 7. Generate samples per class
samples_no  = generate_samples(data_no,  n_no)
samples_yes = generate_samples(data_yes, n_yes)

all_samples = samples_no + samples_yes
np.random.shuffle(all_samples)

# 8. Convert to DataFrame
synthetic_df = pd.DataFrame(all_samples, columns=data.columns)

# 9. Decode categorical columns back
for col, le in label_encoders.items():
    synthetic_df[col] = synthetic_df[col].round().astype(int).clip(0, len(le.classes_) - 1)
    synthetic_df[col] = le.inverse_transform(synthetic_df[col])

# 10. Round numeric columns to match original dtype
int_cols = df.select_dtypes(include=['int64']).columns.tolist()
for col in int_cols:
    synthetic_df[col] = synthetic_df[col].round().astype(int)

# 11. Save file
output_path = "Data/rtd_bank.csv"
try:
    synthetic_df.to_csv(output_path, index=False, sep=';')
    print(f"\n✅ {n_samples} synthetic rows saved to '{output_path}'")
except PermissionError:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"Data/rtd_bank_{timestamp}.csv"
    synthetic_df.to_csv(output_path, index=False, sep=';')
    print(f"\n✅ {n_samples} synthetic rows saved to '{output_path}'")
    print("⚠️  Original file was open — saved with timestamp instead. Close Excel and retry if needed.")

# 12. Print distribution summary
y_counts = synthetic_df[y_col].value_counts()
print(f"\n📊 'y' column distribution:")
print(f"   no  → {y_counts.get('no',  0)} rows ({y_counts.get('no',  0) / n_samples * 100:.1f}%)")
print(f"   yes → {y_counts.get('yes', 0)} rows ({y_counts.get('yes', 0) / n_samples * 100:.1f}%)")