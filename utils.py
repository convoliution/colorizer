import os

import torch
from torchvision import transforms

from PIL import Image


def _resize(img, size):
    """
    Crops and resizes `img` into a square image with side length `size`.

    Parameters
    ----------
    img : PIL.Image.Image
        Image.
    size : int
        Side length for `resized_img`.

    Returns
    -------
    resized_img : PIL.Image.Image
        `img` resized to (`size`, `size`).

    """
    transform = transforms.Compose([transforms.CenterCrop(min(img.size)),
                                    transforms.Resize(size)])
    return transform(img)


def _decolorize(img):
    """
    Converts `img` to single-channel grayscale.

    Parameters
    ----------
    img : PIL.Image.Image
        Image.

    Returns
    -------
    decolorized_img : PIL.Image.Image
        `img` converted to grayscale.

    """
    transform = transforms.Grayscale()
    return transform(img)


def prepare_training_data(source_dir, size):
    """
    Loads images from `source_dir` and preprocesses them into training data tensors.

    Parameters
    ----------
    source_dir : str
        Path to directory containg source images.
    size : int
        Side length for cropping and resizing source images.

    Returns
    -------
    inputs : torch.Tensor of shape (N, 3, `size`, `size`)
        Tensor containing data for N three-channel RGB images.
        Values are in the range [0.0, 1.0].
    targets : torch.Tensor of shape (N, 1, `size`, `size`)
        Tensor containing data for N single-channel grayscale images.
        Values are in the range [0.0, 1.0].

    """
    imgs = [Image.open(os.path.join(source_dir, file_name)).convert('RGB')
            for file_name
            in os.listdir(source_dir)
            if not file_name.startswith('.')]

    target_imgs = [(_resize(img, size)) for img in imgs]
    input_imgs = [(_decolorize(img)) for img in target_imgs]

    targets = torch.stack([transforms.functional.to_tensor(img) for img in target_imgs])
    inputs = torch.stack([transforms.functional.to_tensor(img) for img in input_imgs])

    return inputs, targets


def save(tensor, file_name):
    transforms.functional.to_pil_image(tensor).save(file_name)
