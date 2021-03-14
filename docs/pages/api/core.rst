Core (Catalyst abstractions)
============================

There are 3 main abstractions in `Catalyst <https://catalyst-team.github.io/catalyst>`__: *Experiment*, *Runner*, and *Callback*.

In general, the *Experiment* knows **what** you would like to run, *Runner* contains all the logic of **how** to run
the experiment, while *Callbacks* allow you to customize experiment run logic by using specific custom callback
functions without changing neither *Experiment* nor *Runner*.

.. note::

    To learn more about Catalyst Core concepts, please check out

        - :py:obj:`catalyst.core.experiment.IExperiment` (`docs <https://catalyst-team.github.io/catalyst/api/core.html#catalyst.core.experiment.IExperiment>`__)
        - :py:obj:`catalyst.core.runner.IRunner` (`docs <https://catalyst-team.github.io/catalyst/api/core.html#catalyst.core.runner.IRunner>`__)
        - :py:obj:`catalyst.core.callback.Callback` (`docs <https://catalyst-team.github.io/catalyst/api/core.html#catalyst.core.callback.Callback>`__)


Experiment
^^^^^^^^^^

*Experiment* in an abstraction that contains information about the experiment – a model, a criterion, an optimizer,
a scheduler, and their hyperparameters. It also contains information about the data and transformations used.
In other words, the Experiment knows **what** you would like to run.

.. automodule:: esrgan.core.experiment
    :members:
    :undoc-members:


GAN Runners
^^^^^^^^^^^

*Runner* is an abstraction that knows how to run an experiment. It contains all the logic of **how**
to run the experiment, stages, epoch and batches.

.. automodule:: esrgan.core.runner
    :members:
    :undoc-members:


Callbacks
^^^^^^^^^

*Callback* is an abstraction that lets you customize your experiment run logic.
To give users maximum flexibility and extensibility Catalyst supports callback execution anywhere in the training loop:

.. code:: bash

    -- stage start
    ---- epoch start
    ------ loader start
    -------- batch start
    ---------- batch handler (Runner logic)
    -------- batch end
    ------ loader end
    ---- epoch end
    -- stage end

    exception – if an Exception was raised

For example, to calculate ROC-AUC of the model you may use :py:func:`on_batch_end` method to gather
per-batch predictions and :py:func:`on_loader_end` method to average those statistics.

Metrics
-------

.. automodule:: esrgan.callbacks.metrics
    :members:
    :undoc-members:
