import argparse
import sys

# Импорты наших модулей
from src.data.loader import load_iris_data
from src.features.builder import build_features
from src.model.trainer import train_and_predict
from src.metrics.evaluator import print_report

def parse_args():
    parser = argparse.ArgumentParser(description="ML Pipeline for Iris Dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Доля тестовой выборки (например, 0.3)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed для воспроизводимости")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Запуск ML-пайплайна. Test size: {args.test_size}, Random state: {args.random_state}")
    
    # Шаг 1: Загрузка данных
    print("Загрузка данных...")
    raw_df = load_iris_data()
    
    # Шаг 2: Построение признаков
    print("Построение признаков...")
    df_features = build_features(raw_df)
    
    # Шаг 3: Обучение и предсказание
    print("Обучение модели...")
    model, accuracy, y_pred, y_true = train_and_predict(
        df_features, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    # Шаг 4: Оценка
    print(f"\nAccuracy на тесте: {accuracy:.4f}")
    print_report(y_true, y_pred)
    
    # Здесь можно добавить сохранение модели (joblib.dump) для следующего этапа
    print("\nПайплайн завершён успешно.")

if __name__ == "__main__":
    main()
