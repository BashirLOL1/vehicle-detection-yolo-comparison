from ultralytics import YOLO
import os

DATA_YAML = r"B:\piton\Proektny praktikum\data\Vehicle_Danang_2025\merged_dataset\data.yaml"


def main():
    if not os.path.exists(DATA_YAML):
        print(f"❌ Файл не найден: {DATA_YAML}")
        return

    print(f"✅ Конфиг найден: {DATA_YAML}")
    print("Загрузка YOLO26n...")
    model = YOLO("yolo26n.pt")  # новая модель 2026 года

    print("Начинаем обучение YOLO26n...")
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        name="yolo26n_danang",
        exist_ok=True
    )

    print("✅ Обучение YOLO26n завершено!")


if __name__ == '__main__':
    main()