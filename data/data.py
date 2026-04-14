from typing import Any, List, Tuple
import tensorflow_datasets as tfds
import tensorflow as tf

class Data:
    _CIFAR_10_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self):
        ds = tfds.load("cifar10", split="train", as_supervised=True, data_dir="./data/public") # as_supervised=True means that the dataset is returned as a tuple of (image, label)
        ds = Data._get_cifar_10_ds(ds)
        self.ds = ds

    @staticmethod
    def _get_cifar_10_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
        classes = tf.constant(Data._CIFAR_10_CLASSES, dtype=tf.int64)
        ds = ds.filter(lambda x, y: tf.reduce_any(tf.equal(y, classes)))
        return ds

    def get_dataset(self):
        return self.ds

    def get_batches(self, batch_size: int, number_of_batches: int) -> Tuple[Any]:
        batches = self.ds.shuffle(buffer_size=10000).batch(batch_size).take(number_of_batches)
        return batches

    def get_x_y(self, batch_size: int, number_of_batches: int) -> Tuple[List[Any], List[Any]]:
        batches = self.get_batches(batch_size, number_of_batches)
        x = []
        y = []
        for batch in batches:
            x.append(batch[0])
            y.append(batch[1])
        return x, y