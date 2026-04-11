from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data.loaders import load_vector_dataset


def main():
    # ========================
    # LOAD DATA
    # ========================
    X, y = load_vector_dataset("datasets/email_dataset_github_processed_converted.csv")

    # ========================
    # SPLIT
    # ========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ========================
    # MODEL
    # ========================
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # ========================
    # EVAL
    # ========================
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
