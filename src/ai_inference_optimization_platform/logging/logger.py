import logging

from ai_inference_optimization_platform.config.settings import settings


def setup_logger() -> logging.Logger:
    """
    Configure application logger.
    """

    logger = logging.getLogger("ai_inference_platform")

    if logger.hasHandlers():
        return logger

    logger.setLevel(settings.log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


logger = setup_logger()