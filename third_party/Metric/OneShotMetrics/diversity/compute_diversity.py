import os
from collections import defaultdict

import cv2
import numpy as np


def read_image_as_grayscale(path):
    ref = cv2.imread(path)
    return cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)


def compute_images_diversity(ref_dir, image_dirs):
    """To quantify the diversity of the generated images,
        for each training example we calculated the standard devia-
        tion (std) of the intensity values of each pixel over 100 gen-
        erated images, averaged it over all pixels, and normalized
        by the std of the intensity values of the training image.
    """
    all_diversities = []

    for ref_filename in os.listdir(ref_dir):
        ref_gray = read_image_as_grayscale(os.path.join(ref_dir, ref_filename))

        images = []
        for images_dir in image_dirs:
            images.append(read_image_as_grayscale(os.path.join(images_dir, ref_filename)))
        images = np.stack(images)

        diversity = np.std(images, axis=0).mean() / np.std(ref_gray)
        all_diversities.append(diversity)

    return np.mean(all_diversities)


def compute_images_diversity_files(real_images, fake_images):
    """To quantify the diversity of the generated images,
        for each training example we calculated the standard devia-
        tion (std) of the intensity values of each pixel over 100 gen-
        erated images, averaged it over all pixels, and normalized
        by the std of the intensity values of the training image.
    """

    # all fake image
    images = []
    for fake_image in fake_images:
        fake_gray = fake_image.convert('L')
        images.append(fake_gray)
    images = np.stack(images)

    all_diversities = []
    for real_image in real_images:
        # rgb -> gray
        real_gray = real_image.convert('L')

        # compute diversity
        diversity = np.std(images, axis=0).mean() / np.std(real_gray)
        all_diversities.append(diversity)

    return np.mean(all_diversities)

