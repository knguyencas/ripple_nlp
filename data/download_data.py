from datasets import load_dataset
import pandas as pd

print("Downloading UIT-VSFC dataset...")
ds = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)

print("Dataset loaded!")
print(ds)

train_df = pd.DataFrame(ds['train'])
print("\nTrain sample:")
print(train_df.head())
print("\nColumns:", train_df.columns.tolist())
print("\nLabel distribution:")
print(train_df['sentiment'].value_counts())