import os
from glob import glob


def createDirectoryIfNotExists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def returnClassPaths(className, rootDir="data"):
    classPathJPG = os.path.join(rootDir, className, "*.j pg")
    classPathJPEG = os.path.join(rootDir, className, "*.jpeg")
    classPathPNG = os.path.join(rootDir, className, "*.png")
    classPathBMP = os.path.join(rootDir, className, "*.bmp")
    return glob(classPathJPG) + glob(classPathJPEG) + glob(classPathPNG) + glob(classPathBMP)
