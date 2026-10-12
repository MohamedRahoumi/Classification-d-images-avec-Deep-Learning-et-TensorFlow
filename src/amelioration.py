from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Intel_Image_Classification" / "outputs"
OUTPUT_DIR = DATA_DIR / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "buildings", "forest", "glacier",
    "mountain", "sea", "street"
]

IMAGE_SIZE = (150, 150)
EPOCHS = 30

CONFIGS = [
    {
        "name": "A_baseline",
        "filters": [32, 64, 128],
        "dropout": 0.5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "augmentation": True,
    },
    {
        "name": "B_regularisation",
        "filters": [32, 64, 128],
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "augmentation": True,
    },
    {
        "name": "C_deep",
        "filters": [32, 64, 128, 256],
        "dropout": 0.5,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "augmentation": True,
    },
]


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return image / 255.0, label


def create_dataset(filename, batch_size, training=False, augment=False):
    df = pd.read_csv(DATA_DIR / filename)

    labels = df["class"].map(
        {name: i for i, name in enumerate(CLASSES)}
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, labels.values)
    )

    if training:
        dataset = dataset.shuffle(
            1000, seed=42, reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    if training and augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

        dataset = dataset.map(
            lambda images, labels: (
                augmentation(images, training=True), labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(filters, dropout, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3))
    ])

    for number_filters in filters:
        model.add(tf.keras.layers.Conv2D(
            number_filters, (3, 3), padding="same"
        ))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(6, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


results = []

for config in CONFIGS:
    print(f"\nConfiguration : {config['name']}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    train_ds = create_dataset(
        "train.csv",
        config["batch_size"],
        training=True,
        augment=config["augmentation"]
    )

    validation_ds = create_dataset(
        "validation.csv",
        config["batch_size"]
    )

    model = build_model(
        config["filters"],
        config["dropout"],
        config["learning_rate"]
    )

    model_path = OUTPUT_DIR / f"{config['name']}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True
        ),
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
        callbacks=callbacks,
        verbose=1
    )

    best_epoch = (
        min(
            range(len(history.history["val_loss"])),
            key=history.history["val_loss"].__getitem__
        ) + 1
    )

    best_val_loss = min(history.history["val_loss"])
    best_val_accuracy = max(history.history["val_accuracy"])

    results.append({
        "configuration": config["name"],
        "filters": str(config["filters"]),
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_accuracy": best_val_accuracy,
        "train_accuracy_final": history.history["accuracy"][-1]
    })

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(f"Loss - {config['name']}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title(f"Accuracy - {config['name']}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{config['name']}_curves.png")
    plt.close()

comparison = pd.DataFrame(results).sort_values("val_loss")
comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

print("\nComparaison des configurations :")
print(comparison.to_string(index=False))

best_name = comparison.iloc[0]["configuration"]
best_model = tf.keras.models.load_model(
    OUTPUT_DIR / f"{best_name}.keras"
)
best_model.save(OUTPUT_DIR / "final_model.keras")

print(f"\nModèle sélectionné : {best_name}")
print("Modèle final sauvegardé dans :", OUTPUT_DIR / "final_model.keras")