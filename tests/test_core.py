import unittest

from app.main import main


class TestCoreStartup(unittest.TestCase):
    def test_app_startup(self):
        main()


if __name__ == "__main__":
    unittest.main()
