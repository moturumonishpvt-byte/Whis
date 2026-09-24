import unittest
from app.main import main

class TestAppStartup(unittest.TestCase):
    def test_app_startup(self):
        try:
            main()
            started = True
        except Exception:
            started = False
        self.assertTrue(started)

if __name__ == "__main__":
    unittest.main()
