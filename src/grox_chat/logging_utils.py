import logging


LOG_FORMAT = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with timestamps for runtime processes."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
        )
        return

    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    for handler in root.handlers:
        handler.setFormatter(formatter)
