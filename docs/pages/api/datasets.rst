Datasets
========

The models subpackage contains definitions for the following datasets for image super-resolution:

.. contents::
    :depth: 2
    :local:


All datasets are subclasses of :class:`torch.utils.data.Dataset` i.e, they have ``__getitem__`` and ``__len__`` methods
implemented. Hence, they can all be passed to a :class:`torch.utils.data.DataLoader` which can load multiple samples
parallelly using ``torch.multiprocessing`` workers.
For example: ::

    div2k_data = esrgan.datasets.DIV2KDataset('path/to/div2k_root/')
    data_loader = torch.utils.data.DataLoader(div2k_data, batch_size=4, shuffle=True)


DIV2K
^^^^^

.. autoclass:: esrgan.datasets.DIV2KDataset
    :members:


Flickr2K
^^^^^^^^

.. autoclass:: esrgan.datasets.Flickr2KDataset
    :members:


Folder of Images
^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.datasets.ImageFolderDataset
    :members:
    :undoc-members:
