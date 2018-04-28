from unittest import TestCase

import visdom

from zhai_tools.VisdomGo import VisdomGo
import numpy as np

class TestVisdomGo(TestCase):
    def test_visdom_line(self):
        vg=VisdomGo()
        x = [1, 2, 3, 4, 5, 6]
        y = [21, 21, 65, 13, 43, 12]
        x = np.array(x)
        vis = visdom.Visdom()


