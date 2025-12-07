# metrics_logger.py
import logging

def setup_metrics_logger(level=logging.WARNING):
    """
    Configure the 'metrics' logger used by all metric modules
    (dnsmos, pesq, stoi, sisdr, snr, etc.)

    Call this ONCE at application startup (e.g., in train.py or main.py).
    """
    logger = logging.getLogger("metrics")
    if not logger.handlers:  # prevent duplicate handlers if called multiple times
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
