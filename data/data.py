from typing import Any, List, Tuple
import tensorflow_datasets as tfds
import tensorflow as tf

class CIFAR10Data:
    _CIFAR_10_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, seed: int):
        self.seed = seed
        ds = tfds.load("cifar10", split="train", as_supervised=True, data_dir="./data/public") # as_supervised=True means that the dataset is returned as a tuple of (image, label)
        ds = CIFAR10Data._get_cifar_10_ds(ds)
        self.ds = ds

    @staticmethod
    def _get_cifar_10_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
        classes = tf.constant(CIFAR10Data._CIFAR_10_CLASSES, dtype=tf.int64)
        ds = ds.filter(lambda x, y: tf.reduce_any(tf.equal(y, classes)))
        return ds

    def normalize_x(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(x, dtype=tf.float32) / tf.constant(255.0, dtype=tf.float32)

    def denormalize_x(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(x, dtype=tf.float32) * tf.constant(255.0, dtype=tf.float32)

    def normalize_y(self, y: tf.Tensor) -> tf.Tensor:
        return tf.cast(y, dtype=tf.float32) / tf.constant(len(CIFAR10Data._CIFAR_10_CLASSES), dtype=tf.float32)

    def denormalize_y(self, y: tf.Tensor) -> tf.Tensor:
        return tf.cast(y, dtype=tf.float32) * tf.constant(len(CIFAR10Data._CIFAR_10_CLASSES), dtype=tf.float32)

    def one_hot_encode_y(self, y: tf.Tensor) -> tf.Tensor:
        return tf.one_hot(y, depth=len(CIFAR10Data._CIFAR_10_CLASSES))

    def one_hot_decode_y(self, y: tf.Tensor) -> tf.Tensor:
        return tf.argmax(y, axis=1)

    def get_dataset(self):
        return self.ds

    def get_batches(self, batch_size: int, number_of_batches: int) -> Tuple[Any]:
        batches = self.ds.shuffle(buffer_size=10000, seed=self.seed).batch(batch_size).take(number_of_batches)
        return batches

    def get_x_y(self, batch_size: int, number_of_batches: int) -> Tuple[List[Any], List[Any]]:
        batches = self.get_batches(batch_size, number_of_batches)
        x = []
        y = []
        for batch in batches:
            x.append(self.normalize_x(batch[0]))
            y.append(self.one_hot_encode_y(batch[1]))
        return x, y