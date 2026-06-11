import unittest

from omnivoice.audiobook.offline_audit import audit_offline_runtime
from omnivoice._offline import network_access_allowed


class AudiobookOfflineTest(unittest.TestCase):
    def test_offline_audit_passes_with_defaults(self):
        audit = audit_offline_runtime()

        self.assertTrue(audit.passed, audit.findings)
        self.assertEqual(audit.env["HF_HUB_OFFLINE"], "1")
        self.assertFalse(network_access_allowed())


if __name__ == "__main__":
    unittest.main()
