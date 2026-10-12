import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = "Intel_Image_Classification/outputs"

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0

    return image, label


augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


def create_dataset(filename, training=False):
    df = pd.read_csv(f"{DATA_DIR}/{filename}")

    classes = [
        "buildings", "forest", "glacier",
        "mountain", "sea", "street"
    ]

    labels = df["class"].map(
        {name: i for i, name in enumerate(classes)}
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, labels.values)
    )

    dataset = dataset.map(load_image)

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(BATCH_SIZE)

    if training:
        dataset = dataset.map(
            lambda images, labels: (
                augmentation(images, training=True),
                labels
            )
        )

    return dataset


train_ds = create_dataset("train.csv", training=True)
validation_ds = create_dataset("validation.csv")
test_ds = create_dataset("test.csv")


df = pd.read_csv(f"{DATA_DIR}/train.csv")
path = df.iloc[0]["path"]

image, _ = load_image(path, 0)

image_aug = augmentation(
    tf.expand_dims(image, axis=0),
    training=True
)[0]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image.numpy())
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_aug.numpy())
plt.title("Image augmentée")
plt.axis("off")

plt.show()

for images, labels in train_ds.take(1):
    print("Dimensions :", images.shape)
    print("Pixel minimum :", tf.reduce_min(images).numpy())
    print("Pixel maximum :", tf.reduce_max(images).numpy())


print("Train :", len(pd.read_csv(f"{DATA_DIR}/train.csv")))
print("Validation :", len(pd.read_csv(f"{DATA_DIR}/validation.csv")))
print("Test :", len(pd.read_csv(f"{DATA_DIR}/test.csv") ))