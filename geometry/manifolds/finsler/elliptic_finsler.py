#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 13:36:16 2025

@author: fmry
"""

#%% Modules

from geometry.setup import *

####################

from .manifold import FinslerManifold

#%% Elliptic Finsler

class EllipticFinsler(FinslerManifold):
    def __init__(self,
                 c1:Callable=lambda x,v: 0.75, 
                 c2:Callable=lambda x,v: 0.0,
                 a:Callable=lambda x,v: 1. + (x[0]**2) + (x[1]**2),
                 b:Callable=lambda x,v: 1. + (x[0]**2) + (x[1]**2),
                 theta:Callable=lambda x,v: 0.0,
                 )->None:
        
        self.c1 = c1
        self.c2 = c2
        self.a = a
        self.b = b
        self.theta = theta
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def F_metric(self, x, v):
        
        c1 = self.c1(x,v)
        c2 = self.c2(x,v)
        a = self.a(x,v)
        b = self.b(x,v)
        theta = self.theta(x,v)
        
        x,y = v[0], v[1]
        
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
    
        term1 = x * sin_theta + y * cos_theta
        term2 = x * cos_theta - y * sin_theta
    
        numerator = (
            -a**2 * c2 * term1
            - b**2 * c1 * term2
            + jnp.sqrt(
                a**4 * b**2 * term1**2
                + a**2 * b**4 * term2**2
                - a**2 * b**2 * c1**2 * term1**2
                + 2 * a**2 * b**2 * c1 * c2 * term2 * term1
                - a**2 * b**2 * c2**2 * term2**2
            )
        )
    
        denominator = a**2 * b**2 - a**2 * c2**2 - b**2 * c1**2
    
        return numerator / denominator