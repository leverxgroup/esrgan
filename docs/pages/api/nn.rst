NN
==

These are the basic building block for graphs:

.. contents::
    :depth: 2
    :local:


Containers
----------

ConcatInputModule
^^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.ConcatInputModule
    :members:
    :undoc-members:

ResidualModule
^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.ResidualModule
    :members:
    :undoc-members:


Residual-in-Residual layers
---------------------------

ResidualDenseBlock
^^^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.ResidualDenseBlock
    :members:
    :undoc-members:

ResidualInResidualDenseBlock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.ResidualInResidualDenseBlock
    :members:
    :undoc-members:


UpSampling layers
-----------------

InterpolateConv
^^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.InterpolateConv
    :members:
    :undoc-members:

SubPixelConv
^^^^^^^^^^^^

.. autoclass:: esrgan.nn.SubPixelConv
    :members:
    :undoc-members:


Loss functions
--------------

AdversarialLoss
^^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.AdversarialLoss
    :members:
    :undoc-members:

RelativisticAdversarialLoss
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.RelativisticAdversarialLoss
    :members:
    :undoc-members:

PerceptualLoss
^^^^^^^^^^^^^^

.. autoclass:: esrgan.nn.PerceptualLoss
    :members:
    :undoc-members:


Misc
----

.. autoclass:: esrgan.nn.Conv2dSN
    :members:
    :undoc-members:

.. autoclass:: esrgan.nn.LinearSN
    :members:
    :undoc-members:
