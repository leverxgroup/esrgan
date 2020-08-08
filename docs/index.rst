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
- Mixture of context, perceptual, and adversarial losses. Context and perceptual losses are used for proper image upscaling,
  while adversarial loss pushes neural network to the natural image manifold using a discriminator network
  that is trained to differentiate between the super-resolved images and original photo-realistic images.

.. image:: ./_static/architecture.png


Technologies
============

* `Catalyst` as pipeline runner for deep learning tasks. This new and rapidly developing `library <https://github.com/catalyst-team/catalyst>`__
  can significantly reduce the amount of boilerplate code. If you are familiar with the TensorFlow ecosystem, you can think of Catalyst
  as Keras for PyTorch. This framework is integrated with logging systems such as the well-known `TensorBoard <https://www.tensorflow.org/tensorboard>`__.
* `Pytorch`, `torchvision`, and `PIQ` as main frameworks for deep learning.
* `Albumentations` for data preprocessing.


Quick Start
===========

Setup environment
-----------------

`esrgan` requires python >= 3.8. The `requirements.txt <../requirements.txt>`__ file can be used to install the necessary packages.

::

   git clone  https://github.com/leverxgroup/esrgan.git
   pip install ./esrgan

Run an experiment
-----------------

::

   # step 1 - supervised training of the model
   catalyst-dl run -C esrgan/experiment/config_supervised.yml --benchmark

   # step 2 - use weights from step 1 to train model using GAN approach
   catalyst-dl run -C esrgan/experiment/config_gan.yml --benchmark

where `esrgan/experiment/config.yml` is a path to the `config file <../experiment/config.yml>`__.


Results
=======

Some examples of work of ESRGAN model trained on `DIV2K <https://data.vision.ee.ethz.ch/cvl/DIV2K>`__ dataset:

.. |squirrel_lr| image:: ./_static/sq_crop_lr.png
   :width: 128px
   :height: 128px
.. |squirrel_sr| image:: ./_static/sq_crop_sr_x4.png
   :width: 128px
   :height: 128px
.. |squirrel_hr| image:: ./_static/sq_crop_hr.png
   :width: 128px
   :height: 128px

.. |wolf_lr| image:: ./_static/wf_crop_lr.png
   :width: 128px
   :height: 128px
.. |wolf_sr| image:: ./_static/wf_crop_sr_x4.png
   :width: 128px
   :height: 128px
.. |wolf__hr| image:: ./_static/wf_crop_hr.png
   :width: 128px
   :height: 128px

.. |fish_lr| image:: ./_static/fish_crop_lr.png
   :width: 128px
   :height: 128px
.. |fish_sr| image:: ./_static/fish_crop_sr_x4.png
   :width: 128px
   :height: 128px
.. |fish_hr| image:: ./_static/fish_crop_hr.png
   :width: 128px
   :height: 128px

=====================  ===============  ======================
 LR (low resolution)    ESRGAN (ours)    HR (high resolution)
=====================  ===============  ======================
    |squirrel_lr|       |squirrel_sr|       |squirrel_hr|
      |wolf_lr|           |wolf_sr|           |wolf__hr|
      |fish_lr|           |fish_sr|           |fish_hr|
=====================  ===============  ======================


GitHub
======

The project's GitHub repository can be found `here <https://github.com/leverxgroup/esrgan>`__.
Bugfixes and contributions are very much appreciated!


License
=======

`esrgan` is released under a CC BY-NC-ND 4.0 license. See `LICENSE <../LICENSE>`__ for additional details about it.


.. toctree::
   :maxdepth: 2
   :caption: API

   pages/api/core
   pages/api/models
   pages/api/criterions
   pages/api/datasets
   pages/api/utils

Indices and tables
==================

:ref:`genindex`
