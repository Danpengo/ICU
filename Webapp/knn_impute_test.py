import knn_impute
import data_collection
import unittest
import os
from pathlib import Path



class knn_Test(unittest.TestCase):
    def test_1_standard_case(self):
        x,y = data_collection.collect_data("./testing/test1/test_set","./testing/")
        knn_impute.knn_impute_data(x)

    def test_2_empty_case(self):
        x = Path(os.getcwd())
        os.chdir(x.parent.absolute())
        x,y = data_collection.collect_data("./testing/test3/test_set","./testing/")
        knn_impute.knn_impute_data(x)

    def test_3_mistake_case(self):
        x = Path(os.getcwd())
        os.chdir(x.parent.absolute())
        x,y = data_collection.collect_data("./testing/test5/test_set","./testing/")
        knn_impute.knn_impute_data(x)


unittest.main()
#print(data_collection.collect_data("./testing/test3/test_set","./testing/"))