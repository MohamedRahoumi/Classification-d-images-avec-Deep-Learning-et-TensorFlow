from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Intel_Image_Classification" / "outputs"
MODEL_PATH = DATA_DIR / "model" / "cnn_model_trained.keras"
OUTPUT_DIR = DATA_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "buildings", "forest", "glacier",
    "mountain", "sea", "street"
]

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


df = pd.read_csv(DATA_DIR / "test.csv")

df["label"] = df["class"].map(
    {name: i for i, name in enumerate(CLASSES)}
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (df["path"].values, df["label"].values)
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = df["label"].to_numpy()
probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision :", precision_score(
    y_true, y_pred, average="weighted", zero_division=0
))
print("Recall :", recall_score(
    y_true, y_pred, average="weighted", zero_division=0
))
print("F1-score :", f1_score(
    y_true, y_pred, average="weighted", zero_division=0
))

print("\nRapport par classe :")
print(classification_report(
    y_true,
    y_pred,
    labels=range(len(CLASSES)),
    target_names=CLASSES,
    zero_division=0
))

matrix = confusion_matrix(
    y_true, y_pred, labels=range(len(CLASSES))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=CLASSES
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

df["true_class"] = [CLASSES[i] for i in y_true]
df["predicted_class"] = [CLASSES[i] for i in y_pred]
df["confidence"] = probabilities.max(axis=1)

errors = df[df["label"] != y_pred].copy()
errors.to_csv(OUTPUT_DIR / "misclassified_images.csv", index=False)

print(f"\nImages mal classifiées : {len(errors)} / {len(df)}")

if not errors.empty:
    sample = errors.sample(
        min(12, len(errors)), random_state=42
    )

    plt.figure(figsize=(12, 10))

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        image = tf.keras.utils.load_img(
            row["path"], target_size=IMAGE_SIZE
        )

        plt.subplot(3, 4, i)
        plt.imshow(image)
        plt.title(
            f"Réel : {row['true_class']}\n"
            f"Prédit : {row['predicted_class']}\n"
            f"Confiance : {row['confidence']:.1%}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "misclassified_images.png")
    plt.show()

print("\nÉvaluation terminée.")