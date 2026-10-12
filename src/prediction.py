from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "Intel_Image_Classification" / "outputs" / "experiments" / "final_model.keras"

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

IMAGE_SIZE = (150, 150)

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASSES[predicted_index]
    probability = predictions[0][predicted_index]

    print("Classe prédite :", predicted_class)
    print(f"Probabilité : {probability:.2%}")

    return predicted_class, probability


if __name__ == "__main__":
    image_path = ROOT / "Intel_Image_Classification" / "seg_pred" / "10004.jpg"

    predict_image(image_path)