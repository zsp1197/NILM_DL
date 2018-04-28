# Created by zhai at 2018/1/18
# Email: zsp1197@163.com
import pandas as pd
from scipy.stats import *

class GaussianDiscrete:
    def __init__(self, mean, sigma):
        if(sigma==0):
            sigma=1
            print('sigma 咋是零呢')
        self.interval = 5
        self.pdflist = tuple([norm(mean, sigma).pdf(self.interval * i) for i in range(12 * 60 // self.interval)])
        self.cdflist = tuple([norm(mean, sigma).cdf(self.interval * i) for i in range(12 * 60 // self.interval)])

    def pdf(self, val):
        if int(val) >= self.interval * (len(self.pdflist) - 1): return 1e-8
        if int(val) < 0: return 1e-8
        if int(val) % self.interval == 0:
            return self.pdflist[int(val)//self.interval]
        else:
            a = int(val)//self.interval
            b = int(val) % self.interval
            return self.pdflist[a] + b/self.interval * (self.pdflist[a + 1] - self.pdflist[a])


    def cdf(self, val):
        if int(val) >= self.interval * (len(self.cdflist) - 1): return 1
        if int(val) < 0: return 1e-8
        if int(val) % self.interval == 0:
            return self.cdflist[int(val) // self.interval]
        else:
            a = int(val) // self.interval
            b = int(val) % self.interval
            return self.cdflist[a] + b / self.interval * (self.cdflist[a + 1] - self.cdflist[a])

    def ccdf(self, val):
        return 1 - self.cdf(val)


class Distribution(object):
    def __init__(self,args,type='gaussian_discrete'):
        self.getpdf(args,type)
        self.getcdf(args,type)
        self.args=args


    def getpdf(self, args, type):
        if(type=='gaussian'):
            #TODO std 还是方差？要注意！
            # args=(mean,方差)
            self.pdf=norm(args[0],args[1]).pdf
        elif (type == 'gaussian_discrete'):
            self.pdf=GaussianDiscrete(args[0], args[1]).pdf

    def getcdf(self, args, type):
        if (type == 'gaussian'):
            self.cdf = norm(args[0], args[1]).cdf
        elif (type == 'gaussian_discrete'):
            self.cdf = GaussianDiscrete(args[0], args[1]).cdf


if __name__ == '__main__':
    z = GaussianDiscrete(100,20)
    for k in range(-50, 250, 15):
        print(k, ' ', z.pdf(k),' ', z.cdf(k), ' ', z.ccdf(k))
        k = k - 1.5
        print(k, ' ', z.pdf(k),' ', z.cdf(k), ' ', z.ccdf(k))
    print('')