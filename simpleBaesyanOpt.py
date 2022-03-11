import numpy as np
import math as mt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

class Parameters:
  def __init__(self):
    self.lb = -10
    self.ub = 10
    self.n_iter = 50
    self.xt = np.random.randint(self.lb, self.ub)/2
    self.x0 = np.random.randint(self.lb, self.ub)
    self.ss = np.random.rand()+1
    self.l0 = 1.0
    self.s2 = 0.01
    self.d = 0.1
    self.f = 'm'

def k(x1, x2, l):
  n1 = x1.shape[0]
  n2 = x2.shape[0]
  kk = np.empty((n1, n2))
  for ii in range(0, n1):
    for jj in range(0, n2):
      kk[ii,jj] = np.exp(-((x1[ii]-x2[jj])**2)/(2*l**2))
  return kk

def f0(x, params):
  if params.f == 'm':
    return 2*(1-((x + params.xt)/params.ss)**2)*np.exp(-(x + params.xt)**2/(2*params.ss**2))/(mt.sqrt(3*params.ss)*mt.pi**(1/4))
  if params.f == 'q':  
    return -(x + params.xt)**2

def J(x, alpha, xs, ys, b, params):
  kk = k(x, xs, params.l0)
  return -(kk@alpha@ys + b*(1-np.diag(kk@alpha@np.transpose(kk))))

def ll(x, xs, ys, params):
  kk = k(xs, xs, x[0])
  L = np.linalg.cholesky(kk + params.s2*np.identity(kk.shape[0]))
  alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, ys))
  return np.matmul(np.transpose(ys), alpha) + mt.log(abs(np.linalg.det(kk +  params.s2*np.identity(kk.shape[0]))))

params = Parameters()

dom = np.arange(params.lb,params.ub,0.1)
ys = np.empty((0))
xs = np.empty((0))
xopt = params.x0

for ii in range(0, params.n_iter):
  xs = np.append(xs, [xopt], axis=0)
  ys = np.append(ys, [f0(xopt, params)], axis=0)
  if xs.shape[0] > 10:
    params.l0 = minimize(ll, params.l0, args=(xs, ys, params), method='L-BFGS-B', bounds=Bounds(0.1, 5)).x.item()

  kk = k(xs, xs, params.l0)
  alpha = np.linalg.inv(kk + params.s2*np.eye(np.shape(kk)[0]))
  b = mt.sqrt(2*mt.log((ii+1)**2*mt.pi**2/(6*params.d)))
  x0 = dom[np.argmin(J(dom, alpha, xs, ys, b, params))]
  xopt = minimize(J, x0, args=(alpha, xs, ys, b, params), method='L-BFGS-B', bounds=Bounds(params.lb, params.ub)).x.item()
          
print('Real Optimum: ' + str(-params.xt) + ' Found Optimum: ' + str(xopt) + ' Initial Condition ' + str(params.x0))
plt.plot(dom, f0(dom, params), label='Real Function')
plt.plot(dom, k(dom, xs, params.l0)@alpha@ys, label='Gaussian Process')
plt.plot(dom, J(dom, alpha, xs, ys, b, params), '--', label='Acquisition Function')
plt.plot(xs, ys, '.', label='Sampled Points')
plt.grid()
plt.legend()
plt.show()