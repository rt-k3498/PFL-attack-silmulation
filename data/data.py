from typing import Any, List, Tuple
import tensorflow_datasets as tfds
import tensorflow as tf

class CIFAR10Data:
    _CIFAR_10_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, seed: int, train: bool = True):
        self.seed = seed
        split = "train" if train else "test"
        ds = tfds.load("cifar10", split=split, as_supervised=True, data_dir="./data/public") # as_supervised=True means that the dataset is returned as a tuple of (image, label)
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
    
    def get_uniform_x_y(self, batch_size: int, number_of_batches: int) -> Tuple[List[Any], List[Any]]:
        num_classes = len(CIFAR10Data._CIFAR_10_CLASSES)
        if batch_size % num_classes != 0:
            raise ValueError("batch_size must be a multiple of the number of classes")

        samples_per_class = batch_size // num_classes
        class_batches = []
        for class_label in CIFAR10Data._CIFAR_10_CLASSES:
            class_label_tensor = tf.constant(class_label, dtype=tf.int64)
            class_ds = self.ds.filter(
                lambda _x, y, label=class_label_tensor: tf.equal(tf.cast(y, tf.int64), label)
            )
            class_ds = class_ds.shuffle(
                buffer_size=10000,
                seed=self.seed + class_label,
                reshuffle_each_iteration=False,
            )
            class_batches.append(iter(class_ds.batch(samples_per_class)))

        x = []
        y = []
        for batch_index in range(number_of_batches):
            batch_x = []
            batch_y = []
            for class_batch in class_batches:
                class_x, class_y = next(class_batch)
                batch_x.append(class_x)
                batch_y.append(class_y)

            batch_x = tf.concat(batch_x, axis=0)
            batch_y = tf.concat(batch_y, axis=0)
            indices = tf.random.shuffle(tf.range(tf.shape(batch_y)[0]), seed=self.seed + batch_index)

            x.append(self.normalize_x(tf.gather(batch_x, indices)))
            y.append(self.one_hot_encode_y(tf.gather(batch_y, indices)))
        return x, y
    
    def get_structured_x_y(self, batch_size: int, number_of_batches: int) -> Tuple[List[Any], List[Any]]:
        x = []
        y = []
        num_classes = len(CIFAR10Data._CIFAR_10_CLASSES)

        for batch_index in range(number_of_batches):
            class_label = CIFAR10Data._CIFAR_10_CLASSES[batch_index % num_classes]
            class_label_tensor = tf.constant(class_label, dtype=tf.int64)
            class_ds = self.ds.filter(
                lambda _x, y, label=class_label_tensor: tf.equal(tf.cast(y, tf.int64), label)
            )
            class_ds = class_ds.shuffle(
                buffer_size=10000,
                seed=self.seed + batch_index,
                reshuffle_each_iteration=False,
            )

            batch_x, batch_y = next(iter(class_ds.batch(batch_size).take(1)))
            x.append(self.normalize_x(batch_x))
            y.append(self.one_hot_encode_y(batch_y))

        return x, y
