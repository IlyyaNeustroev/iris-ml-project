from sklearn.datasets import load_iris
import pandas as pd

def load_iris_data():
    """
    Загружает датасет Iris и возвращает DataFrame с признаками и целевым столбцом.
    В будущем здесь будет замена на pandas.read_sql(query, connection) для Oracle.
    """
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df
