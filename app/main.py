import logging
from config.settings import WHIS_ENV, WHIS_DEBUG, WHIS_NAME
from config.logging_config import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(WHIS_NAME)
    
    logger.info("WHIS is starting...")
    logger.info(f"Environment: {WHIS_ENV}")
    logger.info(f"Debug: {WHIS_DEBUG}")
    
    print("WHIS is starting...")
    print("WHIS is ready.")
    
    logger.info("WHIS is ready.")

if __name__ == "__main__":
    main()