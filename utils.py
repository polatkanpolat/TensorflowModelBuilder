import os
from glob import glob
import tensorflow as tf


def createDirectoryIfNotExists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def im_file_to_tensor(file, label, resize=(224, 224)):
    """
    Creates input tensors and labels from image paths 
    """
    def _im_file_to_tensor(file, label):
        path = file
        im = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        im = tf.image.resize(im, size=resize)
        im = tf.cast(im, tf.float32) / 255.0
        return im, label
    return tf.py_function(_im_file_to_tensor,
                          inp=(file, label),
                          Tout=(tf.float32, tf.string))


def returnClassPaths(className, rootDir="data"):
    """
    Returns all image paths for input class name
    """
    classPathJPG = os.path.join(rootDir, className, "*.jpg")
    classPathJPEG = os.path.join(rootDir, className, "*.jpeg")
    classPathPNG = os.path.join(rootDir, className, "*.png")
    classPathBMP = os.path.join(rootDir, className, "*.bmp")
    all_image_paths = glob(classPathJPG) + glob(classPathJPEG) + \
        glob(classPathPNG) + glob(classPathBMP)
    if len(all_image_paths) < 10:
        raise Exception(
            f"there are {len(all_image_paths)} images with class name {className}, but need at least 10 images")
    return all_image_paths


def createDatasetFromClassNames(classNameList):
    all_images = []
    all_labels = []
    for class_name in classNameList:
        class_images = returnClassPaths(class_name)
        all_images.extend(class_images)
        all_labels.extend([class_name for i in range(len(class_images))])

    image_paths = tf.convert_to_tensor(all_images, dtype=tf.string)
    labels = tf.convert_to_tensor(all_labels)

    # Build a TF Queue, shuffle data
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    dataset = dataset.map(im_file_to_tensor)
    return dataset


if __name__ == "__main__":
    dataset = createDatasetFromClassNames(["dog", "cat"])
    image, label = next(iter(dataset))
    print(image.shape)
    print(label.numpy().decode('utf-8'))
