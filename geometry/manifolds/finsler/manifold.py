#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

#%% Riemannian Manifold

class FinslerManifold(ABC):
    def __init__(self,
                 F:Callable[[Array, Array], Array],
                 G:Callable[[Array, Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
                 )->None:
        
        self.F = F
        self.f = f
        self.inv = invf
        
        if  not (G is None):
            self.G = G
            
        return
        
    def __str__(self)->str:
        
        return "Finsler Manifold base object"
    
    def g(self, z:Array, v:Array)->Array:
        
        G = self.G(z,v)
        
        return jnp.einsum('i,ij,j->', v, G, v)
    
    def G(self, z:Array, v:Array)->Array:
        
        return 0.5*jacfwd(lambda v1: grad(lambda v2: self.F(z,v2)**2)(v1))(v)
    
    def Ginv(self, z:Array, v:Array)->Array:
        
        return jnp.linalg.inv(self.G(z,v))
    
    def Dgv(self, z:Array, v:Array)->Array:
        
        return jacfwd(self.gv)(z,v)
    
    def geodesic_equation(self, 
                          z:Array, 
                          v:Array
                          )->Array:
        
        gv = self.gv(z,v)
        Dgv = self.Dgv(z,v)
        
        rhs = jnp.einsum('ikj,i,j->k', Dgv, v, v)-0.5*jnp.einsum('ijk,i,j->k', Dgv, v, v)
        rhs = jnp.linalg.solve(gv, rhs)
        
        dx1t = v
        dx2t = -rhs
        
        return jnp.hstack((dx1t,dx2t))        
    
    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T

        integrand = vmap(lambda g,dg: self.F(g,dg)**2)(gamma[:-1], dgamma)

        return jnp.trapezoid(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T

        integrand = vmap(lambda g,dg: self.F(g,dg))(gamma[:-1],dgamma)
            
        return jnp.trapezoid(integrand, dx=dt)
    
    def indicatrix(self,
                   z:Array,
                   N_points:int=100,
                   *args,
                   )->Array:
        
        theta = jnp.linspace(0.,2*jnp.pi,N_points)
        u = jnp.vstack((jnp.cos(theta), jnp.sin(theta))).T
        
        norm = vmap(self.F, in_axes=(None, 0))(z,u)
        
        return jnp.einsum('ij,i->ij', u, 1./norm)
    