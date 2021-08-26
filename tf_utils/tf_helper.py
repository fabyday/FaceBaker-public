import tensorflow as tf
import os 


def TFcheckpoint_callback_builder(checkpoint_dir, **kwargs):
    
    checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    monitor=kwargs.get("monitor", "val_loss")
    verbose = kwargs.get("verbose", 0)
    save_best_only = kwargs.get("save_best_only", False)
    save_weights_only = kwargs.get("save_weights_only", False)
    mode = kwargs.get("mode", "auto")
    save_freq = kwargs.get("save_freq", "epoch")
    options = kwargs.get("options", None)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor=monitor, 
        verbose=verbose,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode,
        save_freq=save_freq,
        options=options)

    return cp_callback

@tf.function
def mesh_l2_loss(y_true, y_pred):
    """
        y_true, y_pred [ ? vertice_size, dims(3) ]
    """
    sums = tf.keras.backend.sum( tf.keras.backend.pow(y_true - y_pred, 2.0), axis=-1)
    means = tf.keras.backend.mean(sums, -1)
    return tf.keras.backend.mean(means)

