import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    dataframe: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=test_size + val_size,
        stratify=dataframe[label_col],
        random_state=random_state,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        stratify=temp_df[label_col],
        random_state=random_state,
    )

    return train_df, val_df, test_df
