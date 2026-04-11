import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    dataframe: pd.DataFrame,
    label_col: str = "isSpam",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train / validation / test with stratification.

    Args:
        dataframe: Input dataset
        label_col: Name of label column
        test_size: Proportion of test set
        val_size: Proportion of validation set
        random_state: Seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """

    if label_col not in dataframe.columns:
        raise ValueError(f"Column '{label_col}' not found in dataframe")

    # Step 1: split train vs temp (val + test)
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=test_size + val_size,
        stratify=dataframe[label_col],
        random_state=random_state,
    )

    # Step 2: split temp -> val + test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        stratify=temp_df[label_col],
        random_state=random_state,
    )

    return train_df, val_df, test_df
