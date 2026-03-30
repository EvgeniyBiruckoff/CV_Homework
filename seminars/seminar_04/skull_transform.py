import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Загрузка изображения
# -----------------------------
image_path = "data/the_ambassadors.jpg"  # путь к искаженному изображению

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------
# 2. Выбор контрольных точек
# -----------------------------
plt.figure(figsize=(8,8))
plt.imshow(img)
plt.title("Кликните 4 угла искажённого черепа (в порядке: TL, TR, BR, BL)")
points = plt.ginput(4)
plt.close()

src_pts = np.array(points, dtype=np.float32)

# -----------------------------
# 3. Определение целевых точек
# -----------------------------
width = 400
height = 400

dst_pts = np.array([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
], dtype=np.float32)

# -----------------------------
# 4. Расчёт матрицы преобразования
# -----------------------------
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# -----------------------------
# 5. Применение преобразования
# -----------------------------
warped = cv2.warpPerspective(img, M, (width, height))

# -----------------------------
# 6. Показ результата
# -----------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Искажённое изображение")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Исправленный череп")
plt.imshow(warped)
plt.axis("off")

plt.show()

# -----------------------------
# 7. Сохранение результата
# -----------------------------
result = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
cv2.imwrite("corrected_skull.jpg", result)

print("Готово! Сохранено как corrected_skull.jpg")
