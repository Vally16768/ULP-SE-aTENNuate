# si_snr_loss.py
import tensorflow as tf

@tf.function
def si_snr(y_true, y_pred, eps=1e-8):
    # Inputs: [B, T]
    y_true = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    s_target = tf.reduce_sum(y_pred * y_true, axis=-1, keepdims=True) \
               * y_true / (tf.reduce_sum(y_true * y_true, axis=-1, keepdims=True) + eps)

    e_noise  = y_pred - s_target
    si_snr_v = 10.0 * tf.math.log((tf.reduce_sum(s_target*s_target, axis=-1) + eps) /
                                  (tf.reduce_sum(e_noise*e_noise, axis=-1) + eps)) / tf.math.log(10.0)
    return si_snr_v

class SISNRLoss(tf.keras.losses.Loss):
    """Minimize negative SI-SNR (maximize SI-SNR)."""
    def __init__(self, name="si_snr"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    def call(self, y_true, y_pred):
        return -tf.reduce_mean(si_snr(y_true, y_pred))
