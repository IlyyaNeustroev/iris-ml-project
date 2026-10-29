from sklearn.metrics import classification_report

def print_report(y_true, y_pred):
    """
    Выводит подробный отчёт по метрикам.
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
