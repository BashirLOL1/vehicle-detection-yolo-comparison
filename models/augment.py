# import albumentations as A
# import cv2
# import os
#
# INPUT = r"B:\piton\Proektny praktikum\data\raw"
# OUTPUT = r"B:\piton\Proektny praktikum\data\images"
#
# os.makedirs(OUTPUT, exist_ok=True)
#
# aug = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.Rotate(limit=10, p=0.5),
#     A.GaussNoise(p=0.3),
# ])
#
# files = [f for f in os.listdir(INPUT) if f.endswith('.jpg')]
# print(f"Найдено {len(files)} файлов")
#
# for f in files:
#     img = cv2.imread(os.path.join(INPUT, f))
#     if img is None:
#         print(f"❌ Не удалось прочитать {f}")
#         continue
#
#     cv2.imwrite(os.path.join(OUTPUT, f"orig_{f}"), img)
#
#     for i in range(5):
#         aug_img = aug(image=img)['image']
#         cv2.imwrite(os.path.join(OUTPUT, f"aug{i}_{f}"), aug_img)
#
# print(f"✅ Готово! {len(os.listdir(OUTPUT))} картинок в {OUTPUT}")
import os
path = r"B:\piton\Proektny praktikum\data\labels"
print(os.listdir(path))

