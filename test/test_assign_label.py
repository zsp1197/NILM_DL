# Created by zhai at 2018/4/27
from unittest import TestCase
from measure import *
import numpy as np
# Email: zsp1197@163.com
class TestAssign_label(TestCase):
    def test_assign_label(self):
        todeal=np.array([1,2,3,4,4,5,6,6,7,8])
        labels={1.00:0,4.000:1,6.0:2}
        assign_label(todeal=todeal,labels=labels)
