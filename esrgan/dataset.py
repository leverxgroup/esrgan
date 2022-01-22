import glob
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from albumentations.augmentations.crops import functional as F
from catalyst import data
from catalyst.contrib.datasets import misc
import numpy as np
from torch.utils.data import Dataset

from esrgan import utils

__all__ = ["DIV2KDataset", "Flickr2K", "ImageFolderDataset"]


def has_image_extension(uri: Union[str, Path]) -> bool:
    """Check that file has image extension.

    Args:
        uri: The resource to load the file from.

    Returns:
        ``True`` if file has image extension, ``False`` otherwise.

    """
    ext = Path(uri).suffix
    return ext.lower() in {".bmp", ".png", ".jpeg", ".jpg", ".tif", ".tiff"}


def paired_random_crop(
    images: Iterable[np.ndarray], crops_sizes: Iterable[Tuple[int, int]],
) -> Iterable[np.ndarray]:
    """Crop a random part of the input images.

    Args:
        images: Sequence of images.
        crops_sizes: Sequence of crop sizes ``(height, width)``.

    Returns:
        List of crops.

    """
    h_start, w_start = random.random(), random.random()

    crops = [
        F.random_crop(image, height, width, h_start, w_start)
        for image, (height, width) in zip(images, crops_sizes)
    ]

    return crops


class DIV2KDataset(Dataset):
    """`DIV2K <https://data.vision.ee.ethz.ch/cvl/DIV2K>`_ Dataset.

    Args:
        root: Root directory where images are downloaded to.
        train: If True, creates dataset from training set,
            otherwise creates from validation set.
        target_type: Type of target to use, ``'bicubic_X2'``, ``'unknown_X4'``,
            ``'X8'``, ``'mild'``, ...
        patch_size: If ``train == True``, define sizes of patches to produce,
            return full image otherwise. Tuple of height and width.
        transform: A function / transform that takes in dictionary (with low
            and high resolution images) and returns a transformed version.
        low_resolution_image_key: Key to use to store images of low resolution.
        high_resolution_image_key: Key to use to store high resolution images.
        download: If true, downloads the dataset from the internet
            and puts it in root directory. If dataset is already downloaded,
            it is not downloaded again.

    """

    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"
    resources = {
        "DIV2K_train_LR_bicubic_X2.zip": "9a637d2ef4db0d0a81182be37fb00692",
        "DIV2K_train_LR_unknown_X2.zip": "1396d023072c9aaeb999c28b81315233",
        "DIV2K_valid_LR_bicubic_X2.zip": "1512c9a3f7bde2a1a21a73044e46b9cb",
        "DIV2K_valid_LR_unknown_X2.zip": "d319bd9033573d21de5395e6454f34f8",
        "DIV2K_train_LR_bicubic_X3.zip": "ad80b9fe40c049a07a8a6c51bfab3b6d",
        "DIV2K_train_LR_unknown_X3.zip": "4e651308aaa54d917fb1264395b7f6fa",
        "DIV2K_valid_LR_bicubic_X3.zip": "18b1d310f9f88c13618c287927b29898",
        "DIV2K_valid_LR_unknown_X3.zip": "05184168e3608b5c539fbfb46bcade4f",
        "DIV2K_train_LR_bicubic_X4.zip": "76c43ec4155851901ebbe8339846d93d",
        "DIV2K_train_LR_unknown_X4.zip": "e3c7febb1b3f78bd30f9ba15fe8e3956",
        "DIV2K_valid_LR_bicubic_X4.zip": "21962de700c8d368c6ff83314480eff0",
        "DIV2K_valid_LR_unknown_X4.zip": "8ac3413102bb3d0adc67012efb8a6c94",
        "DIV2K_train_LR_x8.zip": "613db1b855721b3d2b26f4194a1d22a6",
        "DIV2K_train_LR_mild.zip": "807b3e3a5156f35bd3a86c5bbfb674bc",
        "DIV2K_train_LR_difficult.zip": "5a8f2b9e0c5f5ed0dac271c1293662f4",
        "DIV2K_train_LR_wild.zip": "d00982366bffee7c4739ba7ff1316b3b",
        "DIV2K_valid_LR_x8.zip": "c5aeea2004e297e9ff3abfbe143576a5",
        "DIV2K_valid_LR_mild.zip": "8c433f812ca532eed62c11ec0de08370",
        "DIV2K_valid_LR_difficult.zip": "1620af11bf82996bc94df655cb6490fe",
        "DIV2K_valid_LR_wild.zip": "aacae8db6bec39151ca5bb9c80bf2f6c",
        "DIV2K_train_HR.zip": "bdc2d9338d4e574fe81bf7d158758658",
        "DIV2K_valid_HR.zip": "9fcdda83005c5e5997799b69f955ff88",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        target_type: str = "bicubic_X4",
        patch_size: Tuple[int, int] = (96, 96),
        transform: Optional[Callable[[Any], Dict]] = None,
        low_resolution_image_key: str = "lr_image",
        high_resolution_image_key: str = "hr_image",
        download: bool = False,
    ) -> None:
        mode = "train" if train else "valid"
        filename_hr = f"DIV2K_{mode}_HR.zip"
        filename_lr = f"DIV2K_{mode}_LR_{target_type}.zip"
        if download:
            # download HR (target) images
            misc.download_and_extract_archive(
                f"{self.url}{filename_hr}",
                download_root=root,
                filename=filename_hr,
                md5=self.resources[filename_hr],
            )

            # download lr (input) images
            misc.download_and_extract_archive(
                f"{self.url}{filename_lr}",
                download_root=root,
                filename=filename_lr,
                md5=self.resources[filename_lr],
            )

        self.train = train

        self.lr_key = low_resolution_image_key
        self.hr_key = high_resolution_image_key

        _, downscaling = target_type.split("_")

        # 'index' files
        lr_images = self._images_in_dir(root, Path(filename_lr).stem)
        hr_images = self._images_in_dir(root, Path(filename_hr).stem)
        assert len(lr_images) == len(hr_images)

        self.data = [
            {"lr_image": lr_image, "hr_image": hr_image}
            for lr_image, hr_image in zip(lr_images, hr_images)
        ]

        self.open_fn = data.ReaderCompose([
            data.ImageReader(input_key="lr_image", output_key=self.lr_key),
            data.ImageReader(input_key="hr_image", output_key=self.hr_key),
        ])

        self.scale = int(downscaling) if downscaling.isdigit() else 4
        height, width = patch_size
        self.target_patch_size = patch_size
        self.input_patch_size = (height // self.scale, width // self.scale)

        self.transform = utils.Augmentor(transform)

    def __getitem__(self, index: int) -> Dict:
        """Gets element of the dataset.

        Args:
            index: Index of the element in the dataset.

        Returns:
            Dict of low and high resolution images.

        """
        record = self.data[index]

        sample_dict = self.open_fn(record)

        if self.train:
            # use random crops during training
            lr_crop, hr_crop = paired_random_crop(
                (sample_dict[self.lr_key], sample_dict[self.hr_key]),
                (self.input_patch_size, self.target_patch_size),
            )
            sample_dict.update({self.lr_key: lr_crop, self.hr_key: hr_crop})

        sample_dict = self.transform(sample_dict)

        return sample_dict

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            Length of the dataset.

        """
        return len(self.data)

    def _images_in_dir(self, *path: Union[str, Path]) -> List[str]:
        # fix path to dir for `NTIRE 2017` datasets
        path = Path(*path)
        if not path.exists():
            idx = path.name.rfind("_")
            path = path.parent / path.name[:idx] / path.name[idx + 1:]

        files = glob.iglob(f"{path}/**/*", recursive=True)
        images = sorted(filter(has_image_extension, files))

        return images


class Flickr2K(DIV2KDataset):
    """`Flickr2K <https://github.com/LimBee/NTIRE2017>`_ Dataset.

    Args:
        root: Root directory where images are downloaded to.
        train: If True, creates dataset from training set,
            otherwise creates from validation set.
        target_type: Type of target to use, ``'bicubic_X2'``, ``'unknown_X4'``,
            ...
        patch_size: If ``train == True``, define sizes of patches to produce,
            return full image otherwise. Tuple of height and width.
        transform: A function / transform that takes in dictionary (with low
            and high resolution images) and returns a transformed version.
        low_resolution_image_key: Key to use to store images of low resolution.
        high_resolution_image_key: Key to use to store high resolution images.
        download: If true, downloads the dataset from the internet
            and puts it in root directory. If dataset is already downloaded,
            it is not downloaded again.

    """

    url = "https://cv.snu.ac.kr/research/EDSR/"
    resources = {
        "Flickr2K.tar": "5d3f39443d5e9489bff8963f8f26cb03",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        target_type: str = "bicubic_X4",
        patch_size: Tuple[int, int] = (96, 96),
        transform: Optional[Callable[[Any], Dict]] = None,
        low_resolution_image_key: str = "lr_image",
        high_resolution_image_key: str = "hr_image",
        download: bool = False,
    ) -> None:
        filename = "Flickr2K.tar"
        if download:
            # download images
            misc.download_and_extract_archive(
                f"{self.url}{filename}",
                download_root=root,
                filename=filename,
                md5=self.resources[filename],
            )

        self.train = train

        self.lr_key = low_resolution_image_key
        self.hr_key = high_resolution_image_key

        degradation, downscaling = target_type.split("_")

        # 'index' files
        subdir_hr = "Flickr2K_HR"
        subdir_lr = Path(f"Flickr2K_LR_{degradation}", downscaling)
        lr_images = self._images_in_dir(root, Path(filename).stem, subdir_hr)
        hr_images = self._images_in_dir(root, Path(filename).stem, subdir_lr)
        assert len(lr_images) == len(hr_images)

        self.data = [
            {"lr_image": lr_image, "hr_image": hr_image}
            for lr_image, hr_image in zip(lr_images, hr_images)
        ]

        self.open_fn = data.ReaderCompose([
            data.ImageReader(input_key="lr_image", output_key=self.lr_key),
            data.ImageReader(input_key="hr_image", output_key=self.hr_key),
        ])

        self.scale = int(downscaling) if downscaling.isdigit() else 4
        height, width = patch_size
        self.target_patch_size = patch_size
        self.input_patch_size = (height // self.scale, width // self.scale)

        self.transform = utils.Augmentor(transform)


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
        images = sorted(filter(has_image_extension, files))

        list_data = [{"image": filename} for filename in images]
        open_fn = data.ReaderCompose([
            data.ImageReader(input_key="image", output_key=image_key),
            data.LambdaReader(input_key="image", output_key=image_name_key),
        ])
        transform = utils.Augmentor(transform)

        super().__init__(
            list_data=list_data, open_fn=open_fn, dict_transform=transform
        )
