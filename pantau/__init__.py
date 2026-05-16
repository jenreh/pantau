import logging

from dotenv import load_dotenv

from pantau.config import configure

logger = logging.getLogger(__name__)
logger.info("Configuring application...")

load_dotenv(override=True)
configuration = configure()
