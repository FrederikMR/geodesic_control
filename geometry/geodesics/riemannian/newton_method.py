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

class SparseNewton(ABC):
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
    
    def hessian_energy(self,
                       zi0:Array,
                       zi1:Array,
                       zi2:Array,
                       )->Array:
        
        zt = jnp.vstack((zi0, zi1, zi2))
        
        terms = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt[:-1])
        vals = jnp.einsum('ti,tij,tj->t', terms, Gt, terms)
        
        return jnp.sum(vals)
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return -grad(lambda z: self.energy(z))(zt)
    
    def Hinner(self,
               zi0:Array,
               zi1:Array,
               zi2:Array,
               argnums:Tuple=(1,2),
               )->Array:
        
        return jacfwd(grad(self.hessian_energy, argnums=argnums[0]), argnums=argnums[1])(zi0, zi1, zi2)
    
    def newton_hessian(self,
                       c:Tuple,
                       y:Tuple,
                       )->Tuple:
        
        si0, si1, Hi1i1, Hi1i2 = c
        si2, zi0, zi1, zi2, zi3 = y
        
        Hi1i2 = self.Hinner(zi0, zi1, zi2, argnums=(1,2))
        Hi2i2 = self.Hinner(zi1, zi2, zi3, argnums=(1,1))
        P = Hi1i2.T
        
        rhs = jnp.concatenate((Hi1i2, jnp.expand_dims(si1, axis=1)), axis=1)
        
        x = jnp.linalg.solve(Hi1i1, rhs)
        
        Hi1i2 = x[:, :self.M.dim]
        si1 = x[:,self.M.dim]

        Hi2i2 -= jnp.einsum('ij,jk->ik', P, Hi1i2)
        si2 -= jnp.einsum('ij,j->i', P, si1)
        
        return ((si1, si2, Hi2i2, Hi1i2),)*2
    
    def newton_update(self,
                      si2:Array,
                      y:Tuple,
                      )->Array:

        si1, Hi1i2 = y

        si1 -= jnp.einsum('ij,j->i', Hi1i2, si2)
        
        return (si1,)*2
        
    
    def newton_step(self,
                    zt:Array,
                    s:Array,
                    )->Array:
        
        zt = jnp.vstack((self.z0, zt, self.zT))
        s = self.Denergy(zt[1:-1])

        z0, z1, z2, z3 = zt[:-3], zt[1:-2], zt[2:-1], zt[3:]
        
        Hi1i1 = self.Hinner(zt[0], zt[1], zt[2], argnums=(1,1))
        Hi1i2 = self.Hinner(zt[0], zt[1], zt[2], argnums=(1,2))

        _, (si0_update, si1_update, Hi1i1, Hi1i2) = lax.scan(self.newton_hessian,
                                                             init=(s[0], s[0], Hi1i1, Hi1i2),
                                                             xs=(s[1:], z0, z1, z2, z3),
                                                             )
        
        si1_last = jnp.linalg.solve(Hi1i1[-1], si1_update[-1])

        _, si1_step = lax.scan(self.newton_update,
                               init = si1_last,
                               xs = (si0_update[::-1], Hi1i2[::-1]),
                               )

        si1_step = jnp.vstack((si1_step[::-1], si1_last))
        
        return si1_step
    
    def update_fun(self,
                   zt:Array,
                   tau:float,
                   s_step:Array,
                   )->Array:
        
        return zt + tau * s_step
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, int],
                 )->Array:
        
        zt, s_grad, idx = carry

        norm_grad = jnp.linalg.norm(s_grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array,Array, int]
                   )->Array:
        
        zt, s_grad, idx = carry

        s_newton = self.newton_step(zt, s_grad)
        
        s_step = lax.cond(jnp.sum(s_grad*s_newton)>0.0,
                          lambda *_: s_newton,
                          lambda *_: s_grad,
                          )
        
        tau = self.line_search(zt, s_step)
        zt = self.update_fun(zt, tau, s_step)
        
        s_grad = self.Denergy(zt)
        
        return (zt, s_grad, idx+1)
    
    def for_step(self,
                 zt:Array,
                 idx:int,
                 )->Array:
        
        s_grad = self.Denergy(zt)
        s_newton = self.newton_step(zt, s_grad)
        
        s_step = lax.cond(jnp.sum(s_grad*s_newton)>0.0,
                          lambda *_: s_newton,
                          lambda *_: s_grad,
                          )
        
        tau = self.line_search(zt, s_step)
        zt = self.update_fun(zt, tau, s_step)
        
        return (zt,)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        zt = self.init_fun(z0,zT,self.T)
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=self.update_fun,
                                            grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=self.update_fun,
                                         **self.line_search_params,
                                         )
        
        self.z0 = z0
        self.zT = zT
        self.G0 = self.M.G(z0)
        
        if step == "while":
            s_grad = self.Denergy(zt)
        
            zt, grad, idx = lax.while_loop(self.cond_fun, 
                                           self.while_step,
                                           init_val=(zt, s_grad, 0)
                                           )
        
            zt = jnp.vstack((z0, zt, zT))
            
        elif step=="for":
            _, val = lax.scan(self.for_step,
                              init=zt,
                              xs=jnp.ones(self.max_iter),
                              )
            zt = val
            
            grad = -vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, grad, idx
    
#%% Gradient Descent Estimation of Geodesics

class SparseRegNewton(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 kappa:float=0.5,
                 lam:float=1.0,
                 line_search_method:str="exact",
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        self.kappa = kappa
        self.lam = lam
        
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
    
    def hessian_energy(self,
                       zi0:Array,
                       zi1:Array,
                       zi2:Array,
                       )->Array:
        
        zt = jnp.vstack((zi0, zi1, zi2))
        
        terms = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt[:-1])
        vals = jnp.einsum('ti,tij,tj->t', terms, Gt, terms)
        
        return jnp.sum(vals)
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return -grad(lambda z: self.energy(z))(zt)
    
    def Hinner(self,
               zi0:Array,
               zi1:Array,
               zi2:Array,
               argnums:Tuple=(1,2),
               )->Array:
        
        return jacfwd(grad(self.hessian_energy, argnums=argnums[0]), argnums=argnums[1])(zi0, zi1, zi2)
    
    def newton_hessian(self,
                       c:Tuple,
                       y:Tuple,
                       )->Tuple:
        
        si0, si1, Hi1i1, Hi1i2, lam = c
        si2, zi0, zi1, zi2, zi3 = y
        
        Hi1i2 = self.Hinner(zi0, zi1, zi2, argnums=(1,2))
        Hi2i2 = self.Hinner(zi1, zi2, zi3, argnums=(1,1)) + lam*jnp.eye(self.M.dim)
        P = Hi1i2.T
        
        rhs = jnp.concatenate((Hi1i2, jnp.expand_dims(si1, axis=1)), axis=1)
        
        x = jnp.linalg.solve(Hi1i1, rhs)
        
        Hi1i2 = x[:, :self.M.dim]
        si1 = x[:,self.M.dim]

        Hi2i2 -= jnp.einsum('ij,jk->ik', P, Hi1i2)
        si2 -= jnp.einsum('ij,j->i', P, si1)
        
        return ((si1, si2, Hi2i2, Hi1i2, lam),)*2
    
    def newton_update(self,
                      si2:Array,
                      y:Tuple,
                      )->Array:

        si1, Hi1i2 = y

        si1 -= jnp.einsum('ij,j->i', Hi1i2, si2)
        
        return (si1,)*2
        
    
    def newton_step(self,
                    zt:Array,
                    s:Array,
                    lam:float,
                    )->Array:
        
        zt = jnp.vstack((self.z0, zt, self.zT))
        s = self.Denergy(zt[1:-1])

        z0, z1, z2, z3 = zt[:-3], zt[1:-2], zt[2:-1], zt[3:]
        
        Hi1i1 = self.Hinner(zt[0], zt[1], zt[2], argnums=(1,1)) + lam*jnp.eye(self.M.dim)
        Hi1i2 = self.Hinner(zt[0], zt[1], zt[2], argnums=(1,2))

        _, (si0_update, si1_update, Hi1i1, Hi1i2, _) = lax.scan(self.newton_hessian,
                                                                init=(s[0], s[0], Hi1i1, Hi1i2, lam),
                                                                xs=(s[1:], z0, z1, z2, z3),
                                                                )
        
        si1_last = jnp.linalg.solve(Hi1i1[-1], si1_update[-1])

        _, si1_step = lax.scan(self.newton_update,
                               init = si1_last,
                               xs = (si0_update[::-1], Hi1i2[::-1]),
                               )

        si1_step = jnp.vstack((si1_step[::-1], si1_last))
        
        return si1_step
    
    def update_fun(self,
                   zt:Array,
                   tau:float,
                   s_step:Array,
                   )->Array:
        
        return zt + tau * s_step
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, int],
                 )->Array:
        
        zt, s_grad, lam, idx = carry

        norm_grad = jnp.linalg.norm(s_grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array,Array, int]
                   )->Array:
        
        zt, s_grad, lam, idx = carry

        s_newton = self.newton_step(zt, s_grad, lam)

        s_step = lax.cond(jnp.sum(s_grad*s_newton)>0.0,
                          lambda *_: s_newton,
                          lambda *_: s_grad,
                          )
        
        tau = self.line_search(zt, s_step)
        zt = self.update_fun(zt, tau, s_step)
        
        s_grad = self.Denergy(zt)
        
        lam *= self.kappa
        
        return (zt, s_grad, lam, idx+1)
    
    def for_step(self,
                 c:Tuple,
                 idx:int,
                 )->Array:
        
        zt, lam = c
        
        s_grad = self.Denergy(zt)
        s_newton = self.newton_step(zt, s_grad, lam)
        
        s_step = lax.cond(jnp.sum(s_grad*s_newton)>0.0,
                          lambda *_: s_newton,
                          lambda *_: s_grad,
                          )
        
        tau = self.line_search(zt, s_step)
        zt = self.update_fun(zt, tau, s_step)
        
        lam *= self.kappa
        
        return (zt,)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        zt = self.init_fun(z0,zT,self.T)
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=self.update_fun,
                                            grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=self.update_fun,
                                         **self.line_search_params,
                                         )
        
        self.z0 = z0
        self.zT = zT
        self.G0 = self.M.G(z0)
        
        if step == "while":
            s_grad = self.Denergy(zt)
            zt, grad, lam, idx = lax.while_loop(self.cond_fun, 
                                                self.while_step,
                                                init_val=(zt, s_grad, self.lam, 0)
                                                )
            zt = jnp.vstack((z0, zt, zT))
            
        elif step=="for":
            _, val = lax.scan(self.for_step,
                              init=(zt, self.lam),
                              xs=jnp.ones(self.max_iter),
                              )
            zt = val
            
            grad = -vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, grad, idx