from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "Intel_Image_Classification" / "outputs" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (150, 150)
NB_CLASSES = 6

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(NB_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.save(OUTPUT_DIR / "cnn_model.keras")

print("Tout il travail bien ")