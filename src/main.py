from src.model import load_data, train_model

if __name__ == "__main__":
    print("Загрузка данных...")
    X, y, feature_names, target_names = load_data()
    print(f"Признаки: {feature_names}")
    print(f"Классы: {target_names}")

    print("Обучение модели...")
    model, acc, X_test, y_test = train_model(X, y)
    print(f"Accuracy на тесте: {acc:.3f}")
