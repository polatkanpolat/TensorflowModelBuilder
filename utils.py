import os
from glob import glob
import tensorflow as tf
from pathlib import Path
import imghdr


def FilterBadImages(class_name, rootDir="data"):
    filepath_list = []
    data_dir = os.path.join(rootDir, class_name)
    # add there all your images file extensions
    image_extensions = [".png", ".jpg"]

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        # print(filepath.name)
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            else:
                filepath_list.append(filepath.name)
    return filepath_list


def createDirectoryIfNotExists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def im_file_to_tensor(file, label):
    """
    Creates input tensors and labels from image paths 
    """
    def _im_file_to_tensor(file, label):
        path = file
        im = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        im = tf.image.resize(im, size=(224, 224))
        im = tf.cast(im, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return im, label
    return tf.py_function(_im_file_to_tensor,
                          inp=(file, label),
                          Tout=(tf.float32, tf.float32))


def returnClassPaths(className, rootDir="data"):
    """
    Returns all image paths for input class name
    """
    all_image_paths = FilterBadImages(className, rootDir=rootDir)

    all_image_paths = [os.path.join(rootDir, className, filepath)
                       for filepath in all_image_paths]

    if len(all_image_paths) < 10:
        raise Exception(
            f"there are {len(all_image_paths)} images with class name {className}, but need at least 10 images")
    return all_image_paths


def createDatasetFromClassNames(classNameList):
    all_images = []
    all_labels = []
    for class_index, class_name in enumerate(classNameList):
        class_images = returnClassPaths(class_name)
        all_images.extend(class_images)
        all_labels.extend([class_index for i in range(len(class_images))])

    image_paths = tf.convert_to_tensor(all_images, dtype=tf.string)
    # labels = tf.convert_to_tensor(all_labels)
    labels = tf.one_hot(all_labels, depth=len(classNameList))
    # Build a TF Queue, shuffle data
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels))

    dataset = dataset.map(im_file_to_tensor)
    return dataset


if __name__ == "__main__":
    dataset = createDatasetFromClassNames(["dog", "cat"])
    image, label = next(iter(dataset))
    print(image.shape)
    print(label.numpy())
    # FilterBadImages("dog")
    # FilterBadImages("cat")
