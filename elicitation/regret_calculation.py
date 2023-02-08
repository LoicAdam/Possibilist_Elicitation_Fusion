# -*- coding: utf-8 -*-
"""Everything to compute the regret."""

import numpy as np
from scipy.optimize import linprog

def pmr_polytope(alternatives, polytope, model):
    """
    Compute the PMR.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    polytope : Polyope
        The Polytope.
    model : Model
        The Model.

    Returns
    -------
    array_like
        PMR.

    """
    nb_alternatives = len(alternatives)
    pmr = np.zeros((nb_alternatives, nb_alternatives))
    poly_a_ub, poly_b_ub, poly_a_eq, poly_b_eq= polytope.get_constrainsts()
    poly_bounds = polytope.get_bounds()
    for i in range(0, nb_alternatives):
        for j in range(0, nb_alternatives):
            if i != j:
                alternatives_diff = model.get_diff(alternatives[j], alternatives[i])
                linprog_res = linprog(c = -alternatives_diff, A_ub = poly_a_ub, b_ub = poly_b_ub, 
                                      A_eq = poly_a_eq, b_eq = poly_b_eq,
                                      bounds = poly_bounds, 
                                      method = 'highs')
                if linprog_res.fun is None:
                    pmr[i,j] = float('inf')
                else:
                    pmr[i,j] = -linprog_res.fun 
    return pmr

def mr_polytope(pmr):
    """
    Compute the MR (from the PMR).

    Parameters
    ----------
    pmr : array_like
        The PMR.

    Returns
    -------
    array_like
        The MR.

    """
    return np.max(pmr, axis = 1)
