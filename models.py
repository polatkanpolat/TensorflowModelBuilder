import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class TFModel:
    def __init__(self, model_name, input_shape, class_nums, weights=None) -> None:
        self.model_name = model_name
        self.input_shape = input_shape
        self.class_nums = class_nums
        self.weights = weights
        self.model = self.GetModel()
        # self.Compile()

    def GetModel(self):
        if self.model_name == "ResNet50":
            return self.GetResnet50()

        elif self.model_name == "VGG16":
            return self.GetVGG16()

        elif self.model_name == "VGG19":
            return self.GetResnet50()

        elif self.model_name == "InceptionV3":
            return self.GetInceptionV3()

        elif self.model_name == "EfficientNetB0":
            return self.GetEfficientNetB0()

        elif self.model_name == "EfficientNetB1":
            return self.GetEfficientNetB1()

        elif self.model_name == "EfficientNetB2":
            return self.GetEfficientNetB2()

        elif self.model_name == "EfficientNetB3":
            return self.GetEfficientNetB3()

        elif self.model_name == "EfficientNetB4":
            return self.GetEfficientNetB4()

        elif self.model_name == "EfficientNetB5":
            return self.GetEfficientNetB5()

        elif self.model_name == "EfficientNetB6":
            return self.GetEfficientNetB6()

        elif self.model_name == "EfficientNetB7":
            return self.GetEfficientNetB7()
        else:
            raise Exception("Model name is not correct")

    def CreateModelFromBase(self, base_model):
        return Sequential(
            [base_model,
             GlobalAveragePooling2D(),
             Dense(128, activation='relu'),
             Dense(self.class_nums, activation='sigmoid')]
        )

    def Compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss='categorical_crossentropy')

    def GetModelInfo(self):
        print(
            f'Model name is {self.model_name} with input shape {self.input_shape} and has {self.class_nums} outputs')
        print(self.model.summary())

    def GetResnet50(self):
        base_model = ResNet50(input_shape=self.input_shape,
                              include_top=False,
                              weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetVGG16(self):
        base_model = VGG16(input_shape=self.input_shape,
                           include_top=False,
                           weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetVGG19(self):
        base_model = VGG19(input_shape=self.input_shape,
                           include_top=False,
                           weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetInceptionV3(self):
        base_model = InceptionV3(input_shape=self.input_shape,
                                 include_top=False,
                                 weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB0(self):
        base_model = EfficientNetB0(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB1(self):
        base_model = EfficientNetB1(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB2(self):
        base_model = EfficientNetB2(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB3(self):
        base_model = EfficientNetB3(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB4(self):
        base_model = EfficientNetB4(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB5(self):
        base_model = EfficientNetB5(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB6(self):
        base_model = EfficientNetB6(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)

    def GetEfficientNetB7(self):
        base_model = EfficientNetB7(input_shape=self.input_shape,
                                    include_top=False,
                                    weights=self.weights)
        return self.CreateModelFromBase(base_model)


if __name__ == "__main__":
    tf_model = TFModel(model_name="InceptionV3",
                       input_shape=(224, 224, 3), class_nums=3)
    tf_model.GetModelInfo()
