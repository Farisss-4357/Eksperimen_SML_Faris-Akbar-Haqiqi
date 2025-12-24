import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(
    file_path,
    output_path,
    train_output_path,
    test_output_path
):
    # 1. Load dataset
    df = pd.read_csv(file_path)
    print(f"Dataset dimuat dari {file_path}")

    # 2. Cek info dataset
    print("\nInfo Dataset:")
    print(df.info())

    # 3. Cek missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # 4. Pisahkan fitur dan target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 5. Scaling fitur numerik (TARGET TIDAK DISENTUH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 6. Gabungkan kembali
    df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    # 7. Simpan dataset preprocessing
    df_processed.to_csv(output_path, index=False)
    print(f"\nData preprocessing disimpan di {output_path}")

    # 8. Train-test split (AMAN)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 9. Simpan train & test
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Data train disimpan di {train_output_path}")
    print(f"Data test disimpan di {test_output_path}")

    return df_processed


if __name__ == "__main__":
    preprocess_data(
        file_path="../namadataset_raw/telecom_churn.csv",
        output_path="telecom_churn_preprocessing.csv",
        train_output_path="train_data.csv",
        test_output_path="test_data.csv"
    )
