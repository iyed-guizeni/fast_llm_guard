import logging

logger = logging.getLogger("fast_llm_guard")
logger.setLevel(logging.ERROR)  # Only ERROR and CRITICAL will be logged

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
if not logger.hasHandlers():
    logger.addHandler(ch)