# Created by zhai at 2017/8/26
# Email: zsp1197@163.com
class Return_class():
    def __init__(self,given):
        self.given=given
        if(isinstance(given,tuple)):
            self._tuple=given
        else:
            raise ValueError('unsupported type '+type(given))

    def __str__(self):
        return str(self.given)

    def get_tuple(self):
        try:
            return self._tuple
        except:
            raise TypeError('no tuple')