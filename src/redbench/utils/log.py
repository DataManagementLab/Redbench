import logging

LOGGER = logging.getLogger()
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)


def log(line, log_mode="info"):
    if log_mode == "debug":
        LOGGER.debug(line)
    elif log_mode == "info":
        LOGGER.info(line)
    elif log_mode == "warning":
        LOGGER.warning(line)
    elif log_mode == "error":
        LOGGER.error(line)
    else:
        raise ValueError(f"Unknown log mode {log_mode}")
