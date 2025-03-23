import gc
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize


# scale an array of images to a new size
def _scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def _calculate_fid(model, images1, images2, batch_size=136):
    act1 = np.empty((len(images1), 2048))
    act2 = np.empty((len(images2), 2048))

    gc.collect()
    for batch_start in range(0, len(images1), batch_size):
        batch_end = batch_start + batch_size
        batch_end = min(batch_end, len(images1))

        act1[batch_start:batch_end] = model.predict(images1[batch_start:batch_end], verbose=0)
        act2[batch_start:batch_end] = model.predict(images2[batch_start:batch_end], verbose=0)

    gc.collect()

    # calculate activations
    # act1 = model.predict(images1, verbose=0)
    # act2 = model.predict(images2, verbose=0)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def _compare_datasets(images1, images2, model):
    # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')

    # resize images
    images1 = _scale_images(images1, (299, 299, 3))
    images2 = _scale_images(images2, (299, 299, 3))

    # pre-process images according to inception v3 expectations
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    fid = _calculate_fid(model, images1, images2)
    return fid


inception_model = None


def compare(dataset1_or_path, dataset2_or_path):
    global inception_model
    if inception_model is None:
        inception_model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    return _compare_datasets(dataset1_or_path, dataset2_or_path, inception_model)

#
# def __getattr__(name):
#
#     if name == "inception_model":
#         return _long_function()