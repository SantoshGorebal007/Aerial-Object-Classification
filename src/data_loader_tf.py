
__all__ = ["get_tf_datasets"]

from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def get_tf_datasets(
    base_dir: str | Path = "../data/classification_dataset",
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
):
    """
    Create normalized TensorFlow datasets for train / valid / test.

    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    base_dir = Path(base_dir)

    # 1) Raw datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir / "train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir / "valid",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir / "test",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="binary",
    )

    class_names = train_ds.class_names

    # 2) Normalization
    def normalize_img(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = (
        train_ds.map(normalize_img, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(normalize_img, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(normalize_img, num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE
    )

    return train_ds, val_ds, test_ds, class_names
