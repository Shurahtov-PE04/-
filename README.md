
#  Разработка сверточной нейронной сети на базе TensorFlow Python для автоматического выявления нарушений ПДД на видеозаписях.

Разработка сверточной нейронной сети на базе TensorFlow Python для автоматического выявления нарушений ПДД по кадрам видеозаписей дорожного трафика.
Ссылка colab: https://colab.research.google.com/drive/12QkV6JwzblqYM_2lI7wlumiTmM5EVGDY?usp=sharing

---

## 1. Постановка задачи

Цель проекта – построить модель компьютерного зрения, которая по отдельному кадру видеозаписи дорожного трафика определяет, относится ли он к:

- **Normal** – обычная дорожная ситуация без нарушений и аварий;
- **Abnormal** – аварийная/аномальная ситуация (столкновения, перевороты и т.п.), связанная с нарушением ПДД.

Нейронная сеть реализована как **бинарный классификатор изображений**. Видеозапись рассматривается как набор кадров; каждый кадр получает метку `normal` или `abnormal`.

---

## 2. Датасет и структура проекта

Данные подготовлены в виде набора кадров, разложенных по папкам:

/train/
abnormal/<clip_id>/.jpg
normal/<clip_id>/.jpg

/val/
abnormal/<clip_id>/.jpg
normal/<clip_id>/.jpg

/test/
abnormal/<clip_id>/.jpg
normal/<clip_id>/.jpg


- Каждый `<clip_id>` соответствует одному исходному видеоролику.
- Внутри клипа файлы названы `1.jpg`, `2.jpg`, … – покадровая развертка видео.
- Класс определяется по верхней папке: `abnormal` → `1`, `normal` → `0`.

Объем данных:

- Train: 7089 кадров  
- Val: 8556 кадров  
- Test: 8694 кадра  

---

## 3. Используемый стек и зависимости

Основные библиотеки:

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- scikit-learn (confusion matrix, classification report, ROC/AUC)  
- seaborn (визуализация матрицы ошибок)

---

## 4. Подготовка данных в коде

1. **Загрузка путей и меток**

Функция `get_all_frames(base_path)` обходит папки `abnormal` и `normal`, собирает все пути к `.jpg` и создает список меток:

0 = normal, 1 = abnormal
label = 1 if class_name == 'abnormal' else 0


В результате получаем списки:

- `train_frames`, `train_labels`
- `val_frames`, `val_labels`
- `test_frames`, `test_labels`

2. **Функция загрузки и предобработки изображения**

def load_image(path, label):
img = tf.io.read_file(path)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, IMG_SIZE) # IMG_SIZE = (128, 128)
img = img / 255.0 # нормализация в​
return img, label


3. **Формирование `tf.data.Dataset`**

train_dataset = tf.data.Dataset.from_tensor_slices((train_frames, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_frames, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_frames, test_labels))

train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

---

## 5. Аугментация данных

Для повышения устойчивости модели к вариациям сцены используется аугментация только на обучающей выборке:

augmentation = keras.Sequential([
layers.RandomFlip("horizontal"),
layers.RandomRotation(0.1),
layers.RandomZoom(0.1),
layers.RandomBrightness(0.2),
layers.RandomContrast(0.2),
])

def augment_image(img, label):
img = augmentation(img, training=True)
return img, label

train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

Далее все датасеты батчатся и оптимизируются:

BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

text

---

## 6. Архитектура нейронной сети

В основе модели — предобученная **MobileNetV2** (на ImageNet), используемая как CNN‑экстрактор признаков:

IMG_SIZE = (128, 128)

base_model = MobileNetV2(
input_shape=(*IMG_SIZE, 3),
include_top=False,
weights='imagenet'
)
base_model.trainable = False

text

Поверх базовой модели строится классификационная «голова»:

model = keras.Sequential([
layers.Input(shape=(*IMG_SIZE, 3)),
base_model,
layers.GlobalAveragePooling2D(),
layers.Dense(256, activation='relu'),
layers.BatchNormalization(),
layers.Dropout(0.5),
layers.Dense(128, activation='relu'),
layers.Dropout(0.4),
layers.Dense(1, activation='sigmoid') # бинарная классификация
])

text

Ключевые элементы архитектуры:

- **MobileNetV2** — свёрточная сеть, предварительно обученная на большом датасете;  
- **GlobalAveragePooling2D** — уменьшает размер признакового пространства и снижает риск переобучения;  
- **Dense + ReLU** — обучаемые слои для адаптации к задаче `normal/abnormal`;  
- **BatchNormalization** и **Dropout** — регуляризация и стабилизация обучения;  
- **Sigmoid‑выход** — выдаёт вероятность `p(abnormal)`.

---

## 7. Обучение модели

Модель компилируется:

LEARNING_RATE = 1e-4

model.compile(
optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
loss='binary_crossentropy',
metrics=['accuracy', keras.metrics.AUC()]
)

Используемые callback’и:

early_stopping = keras.callbacks.EarlyStopping(
monitor='val_loss',
patience=5,
restore_best_weights=True,
verbose=1
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
'/content/best_model.h5',
monitor='val_accuracy',
save_best_only=True,
verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
monitor='val_loss',
factor=0.5,
patience=3,
min_lr=1e-7,
verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

text

Запуск обучения:

EPOCHS = 6

history = model.fit(
train_dataset,
validation_data=val_dataset,
epochs=EPOCHS,
callbacks=callbacks,
verbose=1
)


---

## 8. Оценка и визуализация результатов

### 8.1. Метрики на тестовом наборе

test_loss, test_accuracy, test_auc = model.evaluate(test_dataset, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test AUC:", test_auc)


В одном из запусков были получены значения:

- **Test Accuracy ≈ 0.65**  
- **Test Loss ≈ 0.66**  
- **Test AUC ≈ 0.65**

Это означает, что модель заметно лучше случайного классификатора и способна разделять нормальные и аварийные кадры.

### 8.2. Графики обучения

Построение графиков:

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes.plot(history.history['accuracy'], label='Train Accuracy')
axes.plot(history.history['val_accuracy'], label='Val Accuracy')

axes.plot(history.history['loss'], label='Train Loss')​
axes.plot(history.history['val_loss'], label='Val Loss')​
...
plt.savefig('/content/training_history.png')


> **СЮДА ВСТАВИТЬ ГРАФИКИ accuracy/loss**  
> (вставь `training_history.png` в раздел 8.2 как изображение)

### 8.3. Матрица ошибок и отчёт по классам

y_pred_probs = model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = np.concatenate([y for _, y in test_dataset], axis=0)

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))


Визуализация:

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=['Normal', 'Abnormal'],
yticklabels=['Normal', 'Abnormal'])
plt.savefig('/content/confusion_matrix.png')

<img width="770" height="590" alt="Без названия" src="https://github.com/user-attachments/assets/57c5e13a-4815-4102-916c-7c8d3d755783" />



### 8.4. ROC‑кривая

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot(,, 'k--')​
plt.savefig('/content/roc_curve.png')

text

<img width="789" height="590" alt="Без названия (1)" src="https://github.com/user-attachments/assets/7960767d-5e91-4e3f-ab7b-8f0cafaeb874" />


---

## 9. Сохранение модели и истории обучения

Для дальнейшего использования модель и история сохраняются:

model.save('/content/traffic_model.keras')

history_dict = {
'accuracy': list(map(float, history.history['accuracy'])),
'val_accuracy': list(map(float, history.history['val_accuracy'])),
'loss': list(map(float, history.history['loss'])),
'val_loss': list(map(float, history.history['val_loss'])),
}
with open('/content/training_history.json', 'w') as f:
json.dump(history_dict, f)


---

## 10. Выводы и возможные доработки

Реализован полный pipeline:

- подготовка данных из видеоклипов (кадровый формат),  
- аугментация и формирование `tf.data.Dataset`,  
- сверточная нейросеть на базе MobileNetV2 с головой из Dense‑слоёв, BatchNorm и Dropout,  
- обучение с Adam, binary crossentropy, колбэками EarlyStopping / ModelCheckpoint / ReduceLROnPlateau,  
- оценка качества модели (accuracy, AUC, confusion matrix, ROC).

Дальнейшие направления развития:

- борьба с дисбалансом данных (class weights, oversampling `abnormal`);  
- подбор оптимального порога классификации для повышения полноты по аварийным ситуациям;  
- использование временной информации (последовательность кадров) через 3D‑CNN или рекуррентные сети;  
- дополнительная настройка архитектуры головы и гиперпараметров при наличии GPU.
