import os
import tempfile
import unittest
from unittest.mock import patch

from dronmakr.core import utils


class CollectionsTrashExclusionTests(unittest.TestCase):
    def test_splits_trash_and_archive_excluded_from_collections(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = os.path.join(tmp, "splits")
            os.makedirs(os.path.join(splits, "kick"))
            os.makedirs(os.path.join(splits, "trash"))
            os.makedirs(os.path.join(splits, "archive"))

            open(os.path.join(splits, "kick", "keep.wav"), "wb").close()
            open(os.path.join(splits, "trash", "trashed.wav"), "wb").close()
            open(os.path.join(splits, "archive", "archived.wav"), "wb").close()

            with patch.object(utils, "SPLITS_DIR", splits), patch.object(
                utils, "SAVED_DIR", os.path.join(tmp, "saved")
            ):
                os.makedirs(utils.SAVED_DIR, exist_ok=True)
                files = utils.get_collections_files()

            paths = {item["path"] for item in files}
            self.assertIn("/splits/kick/keep.wav", paths)
            self.assertNotIn("/splits/trash/trashed.wav", paths)
            self.assertNotIn("/splits/archive/archived.wav", paths)


if __name__ == "__main__":
    unittest.main()
