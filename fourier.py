from numpy import *
from numpy.random import default_rng
from scipy.fft import rfft,irfft,rfft2,irfft2,next_fast_len
from scipy.stats import norm
from scipy.special import binom
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.morphology import square,disk,dilation
import matplotlib.pyplot as pl
rn = default_rng()

# imread('../../art/points/Paul_Signac_-_The_Port_of_Rotterdam_-_Google_Art_Project.jpg',as_gray=True)

fd_coeffs = [[.5],[2/3,-1/12],[.75,-.15,1/60],[.8,-.2,4/105,-1/280]]

def findiff_with_err(d,order=2):
    l = len(d[1])
    h = d[0][1]-d[0][0]
    return [sum([(-1)**i*binom(order,i)*d[1][j+order//2-i] for i in range(order+1)]) for j in range(1,l-1)]/h**order

def poisson(n,s,dilate=1,normalize=True):
    m = zeros(s)
    p = rn.integers(zeros([2,n]),[[s[0]]*n,[s[1]]*n])
    m[p[0],p[1]] = 1
    m = dilation(m,square(dilate))
    if normalize: m = m*n/sum(m)
    return m

def sqlattice(s,a,rate=0.,dilate=1):
    x,y = meshgrid(*[arange(0,j,a)+a//2 for j in s])
    x = remainder(x.flatten() + array(norm.rvs(scale=rate*a,size=x.size),int),s[0])
    y = remainder(y.flatten() + array(norm.rvs(scale=rate*a,size=y.size),int),s[1])
    m = zeros(s,int)
    m[x,y] = 1
    return dilation(m,square(dilate))

def stationarized_tiling(img,ns,rate=0.):
    h,w = img.shape
    x,y = range(ns[0]),range(ns[1])
    ry =  array(norm.rvs(scale=rate*h,size=prod(ns)),int).reshape(ns)
    rx =  array(norm.rvs(scale=rate*w,size=prod(ns)),int).reshape(ns)
    return block([[roll(roll(img,ry[i,j],0),rx[i,j],1) for i in x] for j in y])

def volume_fraction_var(i,scale=0.5,every=1,elem=square):
    d = shape(i)
    l = arange(1,int(scale*min(d)),every)
    p = mean(i)
    t = [next_fast_len(x) for x in d]
    f = rfft2(i,t)
    v = array([var(irfft2(f*rfft2(elem(r),t))/r**2) for r in l])
    return array([l,v/p,v/p*sqrt(2/(prod(d)/l**2-1))]) # 1707.01524

def volume_fraction_var_1d(i,scale=0.5,every=1,elem=square):
    d = len(i)
    l = arange(1,int(scale*d),every)
    p = mean(i)
    t = next_fast_len(d)
    f = rfft(i,t)
    v = array([var(irfft(f*rfft(elem(r),t))/r) for r in l])
    return array([l,v/p]) # 1707.01524

def plot_volume_fraction_variance(datas,labels,pre=1.,xmin=None,show=True):
    pl.figure(None,(12,8),facecolor='white')
    if show:
        pl.rc('axes',linewidth=2)
        pl.rc('text',usetex=True)
        pl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    for i,data in enumerate(datas):
        d = data[0].max()
        x = data[0]/d
        pl.errorbar(x,data[1],data[2],label=labels[i],marker='o',mec='k')
    dx = min([data[1,0].min() for data in datas])
    pl.plot(x,pre*dx*x[0]**2/x**2,'--',c='.5',label=r'Poisson $\sim r^{-2}$')
    pl.loglog()
    pl.ylabel(r'Relative volume fraction variance $\sigma^2(r)/\phi$',size=32,labelpad=4)
    pl.xlabel(r'Scaled window radius $r/(\min(w,h)/2)$',size=32)
    pl.grid(True,ls=':',c='.8')
    pl.xticks(fontsize=22)
    pl.yticks(fontsize=22)
    pl.tick_params(pad=10,direction='in')
    ymin = min([data[1].min() for data in datas])
    ymax = max([data[1].max() for data in datas])
    dy = 0.1*(ymax-ymin)
    pl.axis(xmin=xmin,xmax=1,ymin=ymin,ymax=ymax+dy)
    pl.legend(loc='best',fontsize=24)
    pl.tick_params('both',length=12,width=2,top=True,right=True)
    pl.tick_params('both',which='minor',length=8,width=1,direction='in',top=1,right=1)
    fontProperties = {'family':'sans-serif','weight':'normal'}
    a = pl.gca()
    a.set_xticklabels(a.get_xticks(), fontProperties)
    a.set_yticklabels(a.get_yticks(), fontProperties)
    if show:
        pl.show()
        pl.rc('axes',linewidth=1)
        pl.rc('text',usetex=False)
