import os

import tensorflow as tf

import logging
logger = logging.getLogger('hyper_fcn')

class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="epoch", logs=None):
        """Initializer.

        Args:
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        self.iteration = 0
        logs = logs or {}
        if freq not in ["batch", "epoch"]:
            raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = freq
        super(TuneReporter, self).__init__()

    def on_batch_end(self, batch, logs=None):
        from ray import tune
        logs = logs or {}
        if not self.freq == "batch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"))

    def on_epoch_end(self, batch, logs=None):
        from ray import tune
        logs = logs or {}
        if not self.freq == "epoch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs.get("accuracy"))


def create_callbacks(final_run, model_path):
    callbacks = []

    # Creating early stopping callback
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                     patience=10,  
                                                     min_delta=1e-4, 
                                                     mode='min', 
                                                     restore_best_weights=True, 
                                                     verbose=1)
    callbacks.append(earlystopping)

    if final_run:
        logger.info("Creating model checkpoint callback")
        # Make sure the 'snapshots' directory exists
        os.makedirs(model_path, exist_ok=True)

        # Creating model checkpoint callback
        checkpoint_path = model_path
        checkpoint_path = os.path.join(checkpoint_path, 'train_model.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                        monitor='val_loss', 
                                                        save_best_only=True, 
                                                        verbose=1)
        callbacks.append(checkpoint)

    else:
        logger.info("Creating tune reporter callback")
        # Creating ray callback which reports metrics of the ongoing run
        # We choose to report metrics after epoch end using freq="epoch"
        # because val_loss is calculated just before the end of epoch
        tune_reporter = TuneReporter(freq="epoch")

        callbacks.append(tune_reporter)

    return callbacks