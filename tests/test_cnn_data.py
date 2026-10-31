from pathlib import Path
import tempfile
import unittest

import numpy as np

from cnn.data import get_datasets, read_data


class CNNDataTests(unittest.TestCase):
    def test_matrix_labels_padding_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("> 2\n1 2\n3 4\n", encoding="utf-8")
            paths, labels = get_datasets(directory)
            self.assertEqual(paths, [str(path)])
            self.assertEqual(labels, [1])
            np.testing.assert_array_equal(read_data(path, 3, 2)[:, :, 0], [[1, 2], [3, 4], [0, 0]])
            np.testing.assert_array_equal(read_data(path, 1, 2)[:, :, 0], [[1, 2]])
            with self.assertRaisesRegex(ValueError, "columns"):
                read_data(path, 2, 3)


if __name__ == "__main__":
    unittest.main()
