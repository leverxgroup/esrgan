=================================================================
ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
=================================================================

.. raw:: html

    <a class="github-button" href="https://github.com/leverxgroup/esrgan" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star leverxgroup/esrgan on GitHub">Star</a> |
    <a class="github-button" href="https://github.com/leverxgroup/esrgan/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork leverxgroup/esrgan on GitHub">Fork</a> |
    <a class="github-button" href="https://github.com/leverxgroup/esrgan/issues" data-icon="octicon-issue-opened" data-size="large" data-show-count="true" aria-label="Issue leverxgroup/esrgan on GitHub">Issue</a>

----

Pipeine for Image Super-Resolution task that based on a frequently cited paper,
`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks <https://arxiv.org/abs/1809.00219>`__
(Wang Xintao et al.), published in 2018.

In few words, image super-resolution (SR) techniques reconstruct a higher-resolution (HR) image or sequence
from the observed lower-resolution (LR) images, e.g. upscaling of 720p image into 1080p.

One of the common approaches to solving this task is to use deep convolutional neural networks
capable of recovering HR images from LR ones. And ESRGAN (Enhanced SRGAN) is one of them.
Key points of ESRGAN:

- SRResNet-based architecture with residual-in-residual blocks;
- Mixture of context, perceptual, and adversarial losses. Context and perceptual losses are used for proper
  image upscaling, while adversarial loss pushes neural network to the natural image manifold using a discriminator
  network that is trained to differentiate between the super-resolved images and original photo-realistic images.

.. image:: ./_static/architecture.png
   :width: 100%


Technologies
============

* `Catalyst` as pipeline runner for deep learning tasks. This new and rapidly developing `library <https://github.com/catalyst-team/catalyst>`__
  can significantly reduce the amount of boilerplate code. If you are familiar with the TensorFlow ecosystem,
  you can think of Catalyst as Keras for PyTorch. This framework is integrated with logging systems
  such as the well-known `TensorBoard <https://www.tensorflow.org/tensorboard>`__;
* `Pytorch` and `torchvision` as main frameworks for deep learning;
* `Albumentations` and `PIQ` for data processing.


Quick Start
===========

::

   # step 1 - Setup environment, please check `Installation` for more info
   pip install git+https://github.com/leverxgroup/esrgan.git

   # step 2 - Load / prepare config with training details
   wget https://raw.githubusercontent.com/leverxgroup/esrgan/master/config.yml

   # step 3 - train ESRGAN
   catalyst-dl run -C config.yml --benchmark


Results
=======

Some examples of work of ESRGAN model trained on `DIV2K <https://data.vision.ee.ethz.ch/cvl/DIV2K>`__ dataset:

.. table::
   :widths: 25 25 25 25

   +-----------------------------------+---------------------------------+-------------------------------+------------------------------------+
   | .. centered:: LR (low resolution) | .. centered:: `ESRGAN`_         | .. centered:: ESRGAN (ours)   | .. centered:: HR (high resolution) |
   +===================================+=================================+===============================+====================================+
   | .. image:: ./_static/0853lr.png   | .. image:: ./_static/0853sr.png | .. image:: ./_static/0853.png | .. image:: ./_static/0853hr.png    |
   |    :width: 128px                  |    :width: 128px                |    :width: 128px              |    :width: 128px                   |
   |    :height: 128px                 |    :height: 128px               |    :height: 128px             |    :height: 128px                  |
   |    :align: center                 |    :align: center               |    :align: center             |    :align: center                  |
   +-----------------------------------+---------------------------------+-------------------------------+------------------------------------+
   | .. image:: ./_static/0857lr.png   | .. image:: ./_static/0857sr.png | .. image:: ./_static/0857.png | .. image:: ./_static/0857hr.png    |
   |    :width: 128px                  |    :width: 128px                |    :width: 128px              |    :width: 128px                   |
   |    :height: 128px                 |    :height: 128px               |    :height: 128px             |    :height: 128px                  |
   |    :align: center                 |    :align: center               |    :align: center             |    :align: center                  |
   +-----------------------------------+---------------------------------+-------------------------------+------------------------------------+
   | .. image:: ./_static/0887lr.png   | .. image:: ./_static/0887sr.png | .. image:: ./_static/0887.png | .. image:: ./_static/0887hr.png    |
   |    :width: 128px                  |    :width: 128px                |    :width: 128px              |    :width: 128px                   |
   |    :height: 128px                 |    :height: 128px               |    :height: 128px             |    :height: 128px                  |
   |    :align: center                 |    :align: center               |    :align: center             |    :align: center                  |
   +-----------------------------------+---------------------------------+-------------------------------+------------------------------------+

.. _ESRGAN: https://github.com/xinntao/ESRGAN


GitHub
======

The project's GitHub repository can be found `here <https://github.com/leverxgroup/esrgan>`__.
Bugfixes and contributions are very much appreciated!


License
=======

`esrgan` is released under a CC-BY-NC-ND-4.0 license. See `LICENSE <https://github.com/leverxgroup/esrgan/blob/master/LICENSE>`__ for additional details about it.


.. toctree::
   :maxdepth: 3
   :caption: General

   pages/install
   pages/esrgan

.. toctree::
   :maxdepth: 2
   :caption: API

   pages/api/nn
   pages/api/models
   pages/api/datasets
   pages/api/utils
   pages/api/catalyst

Indices and tables
==================

:ref:`genindex`
