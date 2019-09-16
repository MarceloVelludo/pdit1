from skimage.color import rgb2gray


def RGBToGraysScale(image):
    graysScale = rgb2gray(image)
    return graysScale