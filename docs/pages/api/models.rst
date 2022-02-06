Models
======

The models subpackage contains definitions of models for addressing image super-resolution tasks:

.. contents::
    :depth: 2
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

.. autoclass:: esrgan.models.SRResNetEncoder
    :members:
    :undoc-members:

.. autoclass:: esrgan.models.SRResNetDecoder
    :members:
    :undoc-members:


ESRGAN
^^^^^^

.. autoclass:: esrgan.models.ESREncoder
    :members:
    :undoc-members:

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
^^^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.models.StridedConvEncoder
    :members:
    :undoc-members:

LinearHead
^^^^^^^^^^

.. autoclass:: esrgan.models.LinearHead
    :members:
    :undoc-members:
