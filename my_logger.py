import logging


def set_logger(name, file_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(fmt='[%(asctime)s][%(levelname)s] <%(name)s> %(message)s')
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
        if file_name is not None:
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger
