import data_collection
import unittest
import os
from pathlib import Path

class datacollection_Test(unittest.TestCase):
    def test_1_case(self):
        with self.assertRaises(Exception):
            data_collection.collect_data("./testing/test2/test_set","./testing/")

    def test_2_empty_case(self):
        x = Path(os.getcwd())
        x = x.parent.absolute()
        x = x.parent.absolute()
        os.chdir(x.parent.absolute())
        self.assertEqual(len(data_collection.collect_data("./testing/test3/test_set","./testing/")[1]),2)

    def test_3_big_case(self):
        x = Path(os.getcwd())
        os.chdir(x.parent.absolute())
        x = data_collection.collect_data("./testing/test1/test_set","./testing/")

    def test_4_mistake_case(self):
        x = Path(os.getcwd())
        os.chdir(x.parent.absolute())
        x = data_collection.collect_data("./testing/test4/test_set","./testing/")


unittest.main()
