import logging

from dotenv import load_dotenv

from pantau.config import configure
from appkit_commons.configuration.logging import init_logging

# add a basci logging configuration to ensure that logs are visible during configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("Configuring application...")

load_dotenv(override=True)
configuration = configure()
init_logging(configuration)
