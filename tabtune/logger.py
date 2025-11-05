import logging
import logging.handlers
import sys
import os

# Import handlers from the new libraries
from colorlog import ColoredFormatter
from pythonjsonlogger import jsonlogger
from rich.logging import RichHandler


def setup_logger(
    level=logging.INFO,
    log_file=None,
    use_color=True,
    use_rich=False,
    json_format=False,
    max_bytes=5*1024*1024, # 5 MB
    backup_count=5
):
    """
    Configures a sophisticated, configurable logger for the TabTune library.

    Args:
        level (int): The minimum logging level to display.
        log_file (str, optional): Path to a file for log rotation.
        use_color (bool): If True, use colorlog for colored console output.
        use_rich (bool): If True, use rich for advanced console formatting (overrides use_color).
        json_format (bool): If True, format logs as JSON strings.
        max_bytes (int): The maximum size of a log file before rotation.
        backup_count (int): The number of old log files to keep.
    """
    logger = logging.getLogger('tabtune')
    logger.setLevel(level)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Define Formatters ---
    plain_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)-8s] - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    color_formatter = ColoredFormatter(
        '%(log_color)s%(asctime)s - [%(levelname)-8s] - %(name)s:%(reset)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    # --- Configure Console Handler ---
    if use_rich:
        # Rich handler provides coloring, emojis, and beautiful formatting
        console_handler = RichHandler(rich_tracebacks=True, show_path=False)
        console_handler.setFormatter(logging.Formatter('%(message)s')) # Rich handles the rest
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(json_formatter)
        elif use_color:
            console_handler.setFormatter(color_formatter)
        else:
            console_handler.setFormatter(plain_formatter)
    
    logger.addHandler(console_handler)

    # --- Configure File Handler (with rotation) ---
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Use RotatingFileHandler for automatic log file management
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        
        if json_format:
            file_handler.setFormatter(json_formatter)
        else:
            file_handler.setFormatter(plain_formatter)
            
        logger.addHandler(file_handler)

    return logger