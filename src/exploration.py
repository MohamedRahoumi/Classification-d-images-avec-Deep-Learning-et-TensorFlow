from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Intel_Image_Classification" / "seg_train"
OUTPUT_DIR = ROOT / "Intel_Image_Classification" / "outputs"

CLASSES_IMAGE = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

data = []

for class_name in CLASSES_IMAGE:
    for path in (DATA_DIR / class_name).iterdir():
        if not path.is_file():
            continue
        with Image.open(path) as image:
                width, height = image.size
                image.verify()

                data.append({
                    "path": str(path.resolve()),
                    "class": class_name,
                    "width": width,
                    "height": height
                })


df = pd.DataFrame(data)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("total des images:", len(df))
print(df["class"].value_counts())

df["class"].value_counts().reindex(CLASSES_IMAGE, fill_value=0).plot(kind="bar")
plt.savefig(OUTPUT_DIR / "distribution_classes.png")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for ax, class_name in zip(axes.flat, CLASSES_IMAGE):
    row = df[df["class"] == class_name].sample(1, random_state=42).iloc[0]

    with Image.open(row["path"]) as image:
        ax.imshow(image.convert("RGB"))

    ax.set_title(class_name)
    ax.axis("off")

plt.savefig(OUTPUT_DIR / "exemples_images.png")
plt.show()

train, temp = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["class"]
)

validation, test = train_test_split(
    temp, test_size=0.50, random_state=42, stratify=temp["class"]
)

train.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\ntrain :", len(train))
print("validation :", len(validation))
print("test :", len(test))