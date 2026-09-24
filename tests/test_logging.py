import unittest
import logging
import os
from config.logging_config import setup_logging, LOG_FILE

class TestLogging(unittest.TestCase):
    def setUp(self):
        setup_logging()
        self.logger = logging.getLogger("TestLogger")

    def test_logging_initializes(self):
        self.logger.info("Test log entry")
        self.assertTrue(os.path.exists(LOG_FILE))
        
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Test log entry", content)

if __name__ == "__main__":
    unittest.main()