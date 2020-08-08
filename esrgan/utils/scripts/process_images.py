import argparse
import glob
import itertools
import logging
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterator, Optional

from catalyst import utils
import numpy as np

from esrgan.utils import pairwise

logger = logging.getLogger(__name__)


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-workers", "-j", default=1, type=int)
    parser.add_argument(
        "--patch-height", default=512, required=False, type=int
    )
    parser.add_argument(
        "--patch-width", default=None, required=False, type=int
    )
    parser.add_argument(
        "--height-overlap", default=96, required=False, type=int
    )
    parser.add_argument(
        "--width-overlap", default=None, required=False, type=int
    )
    parser.add_argument(
        "--min-height", default=768, required=False, type=int
    )
    parser.add_argument(
        "--min-width", default=None, required=False, type=int
    )

    args = parser.parse_args()
    return args


def cut_with_overlap(
    image: np.ndarray,
    patch_height: int = 512,
    patch_width: int = 512,
    height_overlap: int = 96,
    width_overlap: int = 96,
    min_height: int = 768,
    min_width: int = 768,
) -> Iterator[np.ndarray]:
    """Cut input image with sliding window.

    Args:
        image: Image to cut.
        patch_height: Height of slice to cut.
        patch_width: Width of slice to cut.
        height_overlap: Height of overlap between two slices.
        width_overlap: Width of overlap between two slices.
        min_height: Minimal height that image should have,
            if image is smaller then it wouldn't be cut.
        min_width: Minimal width that image should have,
            if image is smaller then it wouldn't be cut.

    Yields:
        Slice of the image of shape ``patch_height`` x ``patch_width``.

    """
    height, width = image.shape[:2]
    if height > patch_height and width > patch_width:
        x_idxs = range(0, width - patch_width, patch_width - width_overlap)
        y_idxs = range(0, height - patch_height, patch_height - height_overlap)
        patches = itertools.product(pairwise(x_idxs), pairwise(y_idxs))

        for (x_min, x_max), (y_min, y_max) in patches:
            patch = image[y_min:y_max, x_min:x_max]
            yield patch
    else:
        yield image


class Preprocessor:
    def __init__(
        self,
        in_dir: Path,
        out_dir: Path,
        patch_height: int = 512,
        patch_width: Optional[int] = None,
        height_overlap: int = 96,
        width_overlap: Optional[int] = None,
        min_height: int = 768,
        min_width: Optional[int] = None,
    ) -> None:
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.patch_height = patch_height
        self.patch_width = patch_width or patch_height
        self.height_overlap = height_overlap
        self.width_overlap = width_overlap or height_overlap
        self.min_height = min_height
        self.min_width = min_width or min_height

    def preprocess(self, filepath: Path) -> None:
        """Process one file."""
        try:
            image = np.array(utils.imread(filepath))
        except Exception as e:
            logger.warning(f"Cannot read file {filepath}, exception: {e}")
            return

        filename, extention = filepath.stem, filepath.suffix
        out_dir = (self.out_dir / filepath.relative_to(self.in_dir)).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        patches = cut_with_overlap(
            image=image.clip(0, 255).round().astype(np.uint8),
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            height_overlap=self.height_overlap,
            width_overlap=self.width_overlap,
            min_height=self.min_height,
            min_width=self.min_width,
        )

        for index, patch in enumerate(patches):
            out_path = out_dir / f"{filename}_{index}{extention}"
            utils.imwrite(uri=out_path, im=patch)

    def process_all(self, pool: Pool) -> None:
        """Process all images in the folder."""
        files = glob.iglob(f"{self.in_dir}/**/*", recursive=True)
        images = sorted(filter(utils.has_image_extension, files))
        images = [Path(filepath) for filepath in (images)]

        utils.tqdm_parallel_imap(self.preprocess, images, pool)


def main() -> None:
    """Run ``esrgan-process-images`` script."""
    args = args = parse_args().__dict__
    num_workers = args.pop("num_workers")

    with utils.get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    main()
