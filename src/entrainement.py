from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Intel_Image_Classification" / "outputs"
MODEL_DIR = DATA_DIR / "model"
OUTPUT_DIR = DATA_DIR / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30

train_ds = tf.data.Dataset.load(
    str(DATA_DIR / "train_dataset")
)

validation_ds = tf.data.Dataset.load(
    str(DATA_DIR / "validation_dataset")
)

model = tf.keras.models.load_model(
    MODEL_DIR / "cnn_model.keras"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
]

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_DIR / "cnn_model_trained.keras")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Époque")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Époque")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png")
plt.show()

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]
train_loss = history.history["loss"][-1]
val_loss = history.history["val_loss"][-1]

print(f"Train Accuracy : {train_accuracy:.4f}")
print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Train Loss : {train_loss:.4f}")
print(f"Validation Loss : {val_loss:.4f}")
print(f"Meilleure époque : {min(range(len(history.history['val_loss'])), key=history.history['val_loss'].__getitem__) + 1}")