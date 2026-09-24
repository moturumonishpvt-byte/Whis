import os
from dotenv import load_dotenv

load_dotenv()

WHIS_ENV = os.getenv("WHIS_ENV", "development")
WHIS_DEBUG = os.getenv("WHIS_DEBUG", "false").lower() == "true"
WHIS_NAME = os.getenv("WHIS_NAME", "WHIS")