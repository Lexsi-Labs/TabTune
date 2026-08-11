import logging

def setup_logging(logging_level=logging.DEBUG):
    """Setup logging"""
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Clear previous logging configuration
    logging.getLogger().handlers = []

    logging.basicConfig(
        level=logging_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler()]
    )
