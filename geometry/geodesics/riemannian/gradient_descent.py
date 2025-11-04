#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds.riemannian import RiemannianManifold
from geometry.geodesics.line_search import Backtracking, Bisection

#%% Gradient Descent Estimation of Geodesics

class GradientDescent(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 line_search_method:str="exact",
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if line_search_method in ['soft', 'exact']:
            self.line_search_method = line_search_method
        else:
            raise ValueError(f"Invalid value for line search method, {line_search_method}")

        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.z0 = None
        self.G0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Gradient Descent"
    
    def energy(self, 
               zt:Array, 
               *args
               )->Array:
        
        term1 = zt[0]-self.z0
        val1 = jnp.einsum('i,ij,j->', term1, self.G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, int],
                 )->Array:
        
        zt, grad, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array,Array, int]
                   )->Array:
        
        zt, grad, idx = carry
        
        alpha = self.line_search(zt, grad)
        zt -= alpha*grad 
        grad = self.Denergy(zt)
        
        return (zt, grad, idx+1)
    
    def for_step(self,
                 carry:Array,
                 idx:int,
                 )->Array:
        
        zt = carry
        
        grad = self.Denergy(zt)
        alpha = self.line_search(zt, grad)
        zt -= alpha*grad 
        
        return (zt,)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        zt = self.init_fun(z0,zT,self.T)
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=lambda x,alpha,*args: x-alpha*args[0],
                                            grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=lambda x,alpha,*args: x-alpha*args[0],
                                         **self.line_search_params,
                                         )
        
        self.z0 = z0
        self.zT = zT
        self.G0 = self.M.G(z0)
        
        if step == "while":
            grad = self.Denergy(zt)
        
            zt, grad, idx = lax.while_loop(self.cond_fun, 
                                           self.while_step,
                                           init_val=(zt, grad, 0)
                                           )
        
            zt = jnp.vstack((z0, zt, zT))
            
        elif step=="for":
            _, val = lax.scan(self.for_step,
                              init=zt,
                              xs=jnp.ones(self.max_iter),
                              )
            zt = val
            
            grad = vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, grad, idx