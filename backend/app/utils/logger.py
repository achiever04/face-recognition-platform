# backend/app/utils/logger.py

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Basic configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

# Create file handler (rotates daily, keeps 7 days of logs)
log_file = os.path.join(LOG_DIR, "app.log")
file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

# Configure the root logger
def setup_logger(log_level=logging.INFO):
    """Sets up the root logger."""
    root_logger = logging.getLogger()
    
    # Avoid adding handlers multiple times if called again
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Silence overly verbose libraries if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("socketio.server").setLevel(logging.WARNING)
    logging.getLogger("engineio.server").setLevel(logging.WARNING)

# You can get specific loggers in other modules like this:
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is an info message.")

# Call setup_logger early in your application startup (e.g., in main.py)
# setup_logger()