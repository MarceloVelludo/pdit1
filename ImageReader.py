import os

import skimage
from skimage import io


def readImage(path):

    filename = os.path.join(skimage.data_dir, path)
    print(filename)

    return io.imread(filename)