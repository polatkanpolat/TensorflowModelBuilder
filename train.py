import tensorflow as tf
import os
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from datadownloader import ImageDownloader
from utils import createDatasetFromClassNames, createDirectoryIfNotExists
from models import TFModel


class ModelBuilder:
    def __init__(self,
                 experiment_name="",
                 classes=["dog", "cat"],
                 model_name="InceptionV3",
                 size_per_class=100,
                 input_shape=(224, 224, 3),
                 split_size=(0.7, 0.2, 0.1),
                 BATCH_SIZE=4) -> None:
        self.split_size = split_size
        # assert sum(self.split_size) == 1, "split size sum must be 1"
        self.experiment_name = experiment_name
        self.classes = classes
        self.model_name = model_name
        self.size_per_class = size_per_class
        self.input_shape = input_shape
        self.BATCH_SIZE = BATCH_SIZE
        self.model = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.history = None

        self.classes.sort()

    def Prepare(self):
        print("Preparing Builder")
        # self.DownloadImages()
        self.GetDataset()
        self.GetModel()
        self.model.GetModelInfo()
        print("Preparation Completed")

    def DownloadImages(self):
        print("Downloading Images")
        imageDownloader = ImageDownloader()
        for image_class in self.classes:
            imageDownloader.downloadByName(image_class, self.size_per_class)
        print("Images Downloaded")

    def GetDataset(self):
        def prepare_dataset(ds, shuffle=True):
            if shuffle:
                ds = ds.shuffle(buffer_size=500)
            ds = ds.batch(self.BATCH_SIZE)
            # ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds
        print("Prepering Dataset")
        data = createDatasetFromClassNames(self.classes)
        train_size = int(len(data) * self.split_size[0])
        val_size = int(len(data) * self.split_size[1])
        test_size = len(data) - train_size - val_size
        self.train_data = data.take(train_size)
        self.val_data = data.skip(train_size).take(val_size)
        self.test_data = data.skip(train_size+val_size).take(test_size)
        self.train_data = prepare_dataset(self.train_data)
        self.test_data = prepare_dataset(self.test_data)
        self.val_data = prepare_dataset(self.val_data)
        print("Dataset is Ready")

    def GetModel(self):
        print("Preparing Model")
        self.model = TFModel(model_name=self.model_name,
                             input_shape=self.input_shape, class_nums=len(self.classes))
        print("Model is ready")

    def Train(self):
        if self.model is None:
            print("Model has not been created yet, use GetModel method to create")
        else:
            self.model.Compile()
            print("Train is starting ...")
            self.history = self.model.model.fit(
                self.train_data, epochs=2, batch_size=self.BATCH_SIZE, validation_data=self.val_data)
            print("Train Completed")

    def Evaluate(self):
        if self.history is None:
            print("Error!. Model has not been created or trained")
        else:
            print("Evaluation is starting ...")
            acc = CategoricalAccuracy()
            for batch in self.test_data.as_numpy_iterator():
                X, y = batch
                yhat = self.model.model.predict(X)
                acc.update_state(y, yhat)
            print(f"Model test accuracy is {acc.result()}")

    def SaveModel(self):
        print("Model is being saved")
        class_dir = ""
        for class_name in self.classes:
            if class_dir == "":
                class_dir = class_name
            else:
                class_dir += "_vs_" + class_name

        dir_path = os.path.join('models', class_dir)
        createDirectoryIfNotExists(dir_path)
        models_paths = os.listdir(dir_path)
        new_model_path = "experiment_" + \
            self.experiment_name + "_" + self.model_name + "_00.h5"
        num = 1
        while new_model_path in models_paths:
            new_model_path = "experiment_" + self.experiment_name + \
                "_" + self.model_name + "_" + str(num).zfill(2) + ".h5"
            num += 1
        save_path = os.path.join(dir_path, new_model_path)
        self.model.model.save(save_path)
        print(f"Model has been saved to {save_path}")


if __name__ == "__main__":
    builder = ModelBuilder()
    builder.Prepare()
    builder.Train()
    builder.Evaluate()
    builder.SaveModel()
