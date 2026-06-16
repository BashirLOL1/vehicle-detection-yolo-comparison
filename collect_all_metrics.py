from ultralytics import YOLO
import pandas as pd
import os
import multiprocessing as mp

def get_metrics(model_path):
    model = YOLO(model_path)
    metrics = model.val(workers=0)
    return {
        "mAP50": round(metrics.box.map50 * 100, 2),
        "mAP50-95": round(metrics.box.map * 100, 2),
        "Precision": round(metrics.box.mp * 100, 2),
        "Recall": round(metrics.box.mr * 100, 2),
    }

def main():
    models = {
        "YOLOv8n": r"runs/detect/yolov8_danang/weights/best.pt",
        "YOLOv10n": r"runs/detect/yolov10n_danang/weights/best.pt",
        "YOLOv11n": r"runs/detect/yolo11n_danang/weights/best.pt",
        "YOLO26n": r"runs/detect/yolo26n_danang/weights/best.pt",
        "YOLOv11s": r"runs/detect/yolo11s_danang/weights/best.pt",
    }

    params = {
        "YOLOv8n": "3.0M", "YOLOv10n": "2.3M", "YOLOv11n": "2.6M",
        "YOLO26n": "2.4M", "YOLOv11s": "9.4M",
    }
    times = {
        "YOLOv8n": "8 ч", "YOLOv10n": "7 ч", "YOLOv11n": "8 ч",
        "YOLO26n": "7 ч", "YOLOv11s": "5 ч",
    }

    print("=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА МЕТРИК")
    print("=" * 80)

    results = []
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"❌ {name}: файл не найден")
            continue
        print(f"📊 {name}...", end=" ", flush=True)
        try:
            m = get_metrics(path)
            results.append({
                "Модель": name,
                "mAP50 (%)": m["mAP50"],
                "mAP50-95 (%)": m["mAP50-95"],
                "Precision (%)": m["Precision"],
                "Recall (%)": m["Recall"],
                "Параметры": params[name],
                "Время": times[name],
            })
            print(f"✅ mAP50={m['mAP50']}%")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    df.to_csv("models_comparison.csv", index=False)
    print("\n✅ Сохранено: models_comparison.csv")

if __name__ == '__main__':
    mp.freeze_support()
    main()

