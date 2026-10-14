
import pandas as pd

# Membaca dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
metadata = pd.read_csv("metadata.csv")
sample = pd.read_csv("sample_submission.csv")

# Melihat ukuran dataset
print("=== UKURAN DATASET ===")
print("Training:", train.shape)
print("Testing :", test.shape)

# Menampilkan 5 data pertama
print("\n=== DATA TRAINING ===")
print(train.head())

# Melihat informasi kolom
print("\n=== INFORMASI DATA ===")
train.info()

# Mengecek missing values
print("\n=== MISSING VALUES ===")
print(train.isnull().sum())

# Melihat distribusi target
print("\n=== DISTRIBUSI DAMAGE GRADE ===")
print(train["damage_grade"].value_counts().sort_index())
