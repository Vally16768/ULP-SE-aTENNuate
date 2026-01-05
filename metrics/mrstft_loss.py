# mrstft_loss.py
import tensorflow as tf
import numpy as np

def _stft(x, n_fft, hop, win):
    return tf.signal.stft(
        signals=x,
        frame_length=win,
        frame_step=hop,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )

@tf.function
def _spec_convergence(S, S_hat, eps=1e-7):
    num = tf.norm(S - S_hat, ord='fro', axis=[-2, -1])
    den = tf.norm(S,          ord='fro', axis=[-2, -1])
    return tf.reduce_mean(num / (den + eps))

@tf.function
def _log_mag_L1(S, S_hat, eps=1e-7):
    Sm   = tf.math.log(tf.abs(S)   + eps)
    Shat = tf.math.log(tf.abs(S_hat)+ eps)
    return tf.reduce_mean(tf.abs(Sm - Shat))

class MRSTFTLoss(tf.keras.losses.Loss):
    """
    Multi-Resolution STFT loss:
      loss = mean_over_resolutions( alpha*SC + beta*logMagL1 )
    Inputs expected as waveforms in [-1,1] with shape [B, T].
    """
    def __init__(self,
                 fft_sizes=(256, 512, 1024),
                 hops=(64, 128, 256),
                 wins=(256, 512, 1024),
                 alpha=0.5,
                 beta=0.5,
                 name="mrstft"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.fft_sizes = fft_sizes
        self.hops = hops
        self.wins = wins
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        # y_*: [B, T]
        losses = []
        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.wins):
            S     = _stft(y_true, n_fft, hop, win)
            S_hat = _stft(y_pred, n_fft, hop, win)
            sc  = _spec_convergence(S, S_hat)
            l1  = _log_mag_L1(S, S_hat)
            losses.append(self.alpha*sc + self.beta*l1)
        return tf.add_n(losses) / float(len(losses))
