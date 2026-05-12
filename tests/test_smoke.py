import unittest

class TestSmoke(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    # Stub. Returns True always.
    def test_smoke(self):
        self.assertTrue(True)