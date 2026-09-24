import unittest
from config.settings import WHIS_ENV, WHIS_DEBUG, WHIS_NAME

class TestConfig(unittest.TestCase):
    def test_environment_variables(self):
        self.assertIsNotNone(WHIS_ENV)
        self.assertIsInstance(WHIS_DEBUG, bool)
        self.assertIsNotNone(WHIS_NAME)

if __name__ == "__main__":
    unittest.main()
