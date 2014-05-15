#!/usr/bin/env python
"""
@package ion_functions.data.fdc_functions
@file ion_functions/data/fdc_functions.py
@author Craig Risien and Russell Desiderio
@brief Module containing FDC related data-calculations.
"""

import numpy as np


def fdc_fdchp_grv(lat):
    """
    Description:

        Calculates gravity (acceleration due to earth's gravitational field)
        as a function of latitude. This code is straight from the FDCHP DPS.

    Implemented by:

        2014-05-15: Russell Desiderio. Initial Code

    Usage:

        g = fdc_fdchp_grv(lat)

            where

        g = acceleration due to earth's gravitational field [m/s/s]
        lat = latitude of instrument in decimal degrees.

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """
    # constants from the DPS
    gamma = 9.7803267715     # equatorial value for 'g'
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007

    x = np.sin(np.radians(lat))
    xsq = x * x

    g = gamma * (1.0 + xsq * (c1 + xsq * (c2 + xsq * (c3 + xsq * c4))))
    return g
