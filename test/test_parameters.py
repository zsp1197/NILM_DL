# Created by zhai at 2017/11/15
from unittest import TestCase


# Email: zsp1197@163.com
from Parameters import Parameters


class TestParameters(TestCase):
    def test_get_sax_step(self):
        para=Parameters()
        steps=para.get_sax_step()
        print(steps)
