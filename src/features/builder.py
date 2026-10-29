import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит дополнительные признаки.
    Сейчас возвращает данные как есть, но сюда можно добавить логику.
    """
    # Пример будущей логики:
    # df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
    return df.copy()
