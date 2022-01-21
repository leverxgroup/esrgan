import glob
from typing import Callable, Dict, Optional

from catalyst import data, utils

from esrgan.dataset import misc

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(data.ListDataset):
    """A generic data loader where the samples are arranged in this way: ::

        <pathname>/xxx.ext
        <pathname>/xxy.ext
        <pathname>/xxz.ext
        ...
        <pathname>/123.ext
        <pathname>/nsdf3.ext
        <pathname>/asd932_.ext

    Args:
        pathname: Root directory of dataset.
        image_key: Key to use to store image.
        image_name_key: Key to use to store name of the image.
        transform: A function / transform that takes in dictionary
            and returns its transformed version.

    """

    def __init__(
        self,
        pathname: str,
        image_key: str = "image",
        image_name_key: str = "filename",
        transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        files = glob.iglob(pathname, recursive=True)
        images = sorted(filter(utils.has_image_extension, files))

        list_data = [{"image": filename} for filename in images]
        open_fn = data.ReaderCompose([
            data.ImageReader(input_key="image", output_key=image_key),
            data.LambdaReader(input_key="image", output_key=image_name_key),
        ])
        transform = misc.Augmentor(transform)

        super().__init__(
            list_data=list_data, open_fn=open_fn, dict_transform=transform
        )
