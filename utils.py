import os


def createDirectoryIfNotExists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
