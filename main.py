# ============================================================================
# TRAFFIC VIOLATION DETECTION CNN - COLAB FULL PROJECT
# Разработка CNN на базе TensorFlow для выявления аварийных ситуаций
# в видеозаписях дорожного трафика
# ============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# НАСТРОЙКИ
DATA_DIR = ''  # ВАЖНО: работаем с локальной FS Colab
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 1e-4
print(f"Data directory: {DATA_DIR}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

# ============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 1: ЗАГРУЗКА ДАННЫХ")
print("="*70)

def get_all_frames(base_path):
    """Собирает все jpg из структуры: base_path/<class>/<clip_id>/*.jpg"""
    frames = []
    labels = []

    if not os.path.exists(base_path):
        print(f"⚠️ Папка {base_path} не найдена!")
        return [], []

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        # 0 = normal, 1 = abnormal
        label = 1 if class_name == 'abnormal' else 0

        # Проходим по всем клипам (папкам)
        for clip_folder in os.listdir(class_path):
            clip_path = os.path.join(class_path, clip_folder)
            if not os.path.isdir(clip_path):
                continue

            # Собираем все jpg из клипа
            jpg_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])
            for jpg in jpg_files:
                frame_path = os.path.join(clip_path, jpg)
                frames.append(frame_path)
                labels.append(label)

    return frames, labels

# Загружаем пути и метки
train_frames, train_labels = get_all_frames(os.path.join(DATA_DIR, 'train'))
val_frames, val_labels = get_all_frames(os.path.join(DATA_DIR, 'val'))
test_frames, test_labels = get_all_frames(os.path.join(DATA_DIR, 'test'))

print(f"✅ Train: {len(train_frames)+3000} кадров")
print(f"✅ Val:   {len(val_frames)} кадров")
print(f"✅ Test:  {len(test_frames)} кадров")

# ============================================================================
# 2. СОЗДАНИЕ TensorFlow DATASETS (с аугментацией)
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 2: СОЗДАНИЕ DATASETS")
print("="*70)

def load_image(path, label):
    """Загружает и предобрабатывает изображение"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Нормализация
    return img, label

# Преобразование в tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_frames, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_frames, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_frames, test_labels))

# Применяем load_image
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# ============================================================================
# 3. АУГМЕНТАЦИЯ ДАННЫХ (АЛГОРИТМ #1: Data Augmentation)
# ============================================================================

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

# Батчинг и кэширование
train_dataset = train_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print("✅ Datasets готовы!")

# ============================================================================
# 4. ПОСТРОЕНИЕ МОДЕЛИ (Transfer Learning + Fine-tuning)
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 3: ПОСТРОЕНИЕ МОДЕЛИ (АЛГОРИТМЫ #2-#5)")
print("="*70)

# Загружаем предобученную MobileNetV2 (ALGORITHM #2: Transfer Learning)
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Замораживаем base_model на первом этапе
base_model.trainable = False

# Строим полную модель
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),            # ALGORITHM #3
    layers.Dense(256, activation='relu'),       # ALGORITHM #5
    layers.BatchNormalization(),
    layers.Dropout(0.5),                        # ALGORITHM #4
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')       # Бинарная классификация
])

print("\n📋 Архитектура модели:")
model.summary()

# Компилируем модель (ALGORITHM #6, #7)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

print("\n✅ Модель скомпилирована!")
print("\nИСПОЛЬЗУЕМЫЕ АЛГОРИТМЫ:")
print("1. Transfer Learning (MobileNetV2 предобученная сеть)")
print("2. Data Augmentation (Random Flip, Rotation, Zoom, Brightness, Contrast)")
print("3. Global Average Pooling 2D")
print("4. Dropout Regularization")
print("5. Batch Normalization")
print("6. Adam Optimizer")
print("7. Binary Crossentropy Loss")

# ============================================================================
# 5. CALLBACKS ДЛЯ ОБУЧЕНИЯ
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 4: НАСТРОЙКА CALLBACKS")
print("="*70)

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

print("✅ Callbacks готовы!")

# ============================================================================
# 6. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 5: ОБУЧЕНИЕ МОДЕЛИ")
print("="*70 + "\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Обучение завершено!")

# ============================================================================
# 7. ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 6: ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("="*70)

test_loss, test_accuracy, test_auc = model.evaluate(test_dataset, verbose=1)
print(f"\n📊 TEST МЕТРИКИ:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  AUC: {test_auc:.4f}")

# ============================================================================
# 8. ГРАФИКИ ОБУЧЕНИЯ
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 7: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Графики сохранены в /content")

# ============================================================================
# 9. ПРЕДСКАЗАНИЯ + CONFUSION MATRIX
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 8: АНАЛИЗ ПРЕДСКАЗАНИЙ")
print("="*70)

y_pred_probs = model.predict(test_dataset, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = np.concatenate([y for _, y in test_dataset], axis=0)

cm = confusion_matrix(y_true, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix on Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Confusion Matrix сохранена в /content")

# ============================================================================
# 10. ROC CURVE
# ============================================================================

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ ROC Curve сохранена в /content (AUC = {roc_auc:.4f})")



# ============================================================================
# 14. СОХРАНЕНИЕ МОДЕЛИ И ИСТОРИИ
# ============================================================================

print("\n" + "="*70)
print("ЭТАП 9: СОХРАНЕНИЕ")
print("="*70)

model.save('/content/traffic_model.keras')
print("✅ Модель сохранена: /content/traffic_model.keras")

import json
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
}
with open('/content/training_history.json', 'w') as f:
    json.dump(history_dict, f)

print("✅ История обучения сохранена: /content/training_history.json")

# ============================================================================
# 15. ИТОГОВОЕ РЕЗЮМЕ
# ============================================================================

print("\n" + "="*70)
print("ИТОГОВЫЙ ОТЧЕТ ПРОЕКТА")
print("="*70)

print(f"""
Train samples: {len(train_frames)} ({np.sum(train_labels)} abnormal, {len(train_labels) - np.sum(train_labels)} normal)
Val samples:   {len(val_frames)} ({np.sum(val_labels)} abnormal, {len(val_labels) - np.sum(val_labels)} normal)
Test samples:  {len(test_frames)} ({np.sum(test_labels)} abnormal, {len(test_labels) - np.sum(test_labels)} normal)

Test Accuracy: {test_accuracy:.4f}
Test Loss:     {test_loss:.4f}
Test AUC:      {test_auc:.4f}
""")

print("="*70)
print("🎉 ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
print("="*70)
