Models
======

The models subpackage contains definitions of models for addressing image super-resolution tasks:

.. toctree::
   :titlesonly:

.. contents::
   :local:


Generators
----------

EncoderDecoderNet
^^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.models.EncoderDecoderNet
    :members:
    :undoc-members:


SRGAN
^^^^^

SRResNetEncoder
~~~~~~~~~~~~~~~

.. autoclass:: esrgan.models.SRResNetEncoder
    :members:
    :undoc-members:

SRResNetDecoder
~~~~~~~~~~~~~~~

.. autoclass:: esrgan.models.SRResNetDecoder
    :members:
    :undoc-members:


ESRGAN
^^^^^^

ESREncoder
~~~~~~~~~~

.. autoclass:: esrgan.models.ESREncoder
    :members:
    :undoc-members:

ESRNetDecoder
~~~~~~~~~~~~~

.. autoclass:: esrgan.models.ESRNetDecoder
    :members:
    :undoc-members:


Discriminators
--------------

VGGConv
^^^^^^^

.. autoclass:: esrgan.models.VGGConv
    :members:
    :undoc-members:

StridedConvEncoder
~~~~~~~~~~~~~~~~~~

.. autoclass:: esrgan.models.StridedConvEncoder
    :members:
    :undoc-members:

LinearHead
~~~~~~~~~~

.. autoclass:: esrgan.models.LinearHead
    :members:
    :undoc-members:
