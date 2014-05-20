#!/usr/bin/env python
"""
@package ion_functions.data.fdc_functions
@file ion_functions/data/fdc_functions.py
@author Craig Risien and Russell Desiderio
@brief Module containing FDC related data-calculations.
"""

import numpy as np
import scipy as sp


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
    # equatorial value for 'g'
    gamma = 9.7803267715
    # coefficients of polynomial in (sin(lat))^2
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007

    x = np.sin(np.radians(lat))
    xsq = x * x

    # Horner's method for calculating polynomials
    g = gamma * (1.0 + xsq * (c1 + xsq * (c2 + xsq * (c3 + xsq * c4))))
    return g


def fdc_fdchp_alignwind(u):
    """
    Description:

        Rotates wind velocity components into the streamwise wind. This code is straight from
        the FDCHP DPS.

    Implemented by:

        2014-05-19: Russell Desiderio. Initial Code; converted arithmetic to matrix multiplication.

    Usage:

        [u_rot] = fdc_fdchp_alignwind(u)

            where

        u_rot = (3xL) wind velocity rotated into the "streamwise" coordinate system
        u = (3XL) wind velocity in the earth frame (uncorrected for magnetic declination)

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """
    # mean wind velocity components
    u_mean = np.mean(u, axis=-1)

    # calculate angles for coordinate rotation
    uhor = np.sqrt(u_mean[0] * u_mean[0] + u_mean[1] * u_mean[1])
    beta = np.atan2(u_mean[2], uhor)
    alpha = np.atan2(u_mean[1], u_mean[0])

    # populate rotation matrix
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)
    R = np.array([[ cos_a*cos_b,  sin_a*cos_b, sin_b],
                  [   -sin_a,        cos_a,      0  ],
                  [-cos_a*sin_b, -sin_a*sin_b, cos_b]])

    # the Einstein summation is here configured to do the matrix
    # multiplication u_rot(i,k) = R(i,j) * u(j,k).
    u_rot = np.einsum('ij,jk->ik', R, u)

    return u_rot


def fdc_fdchp_accelsclimode(bhi, ahi, sf, accm, euler, L):
    """
    Description:

        Rotates linear accelerations measured on the FDCHP platform into an earth reference
        system, then integrates to get platform velocity and displacement. This code is
        straight from the FDCHP DPS.

    Implemented by:

        2014-05-19: Russell Desiderio. Initial Code

    Usage:

        [acc, uvwplat, xyzplat] = fdc_fdchp_accelsclimode(bhi, ahi, sf, accm, euler, L)

            where

        acc = (3xL) linear accelerations in "FLIP/Earth" reference
        uvwplat = (3xL) linear velocities at the point of measurement
        xyzplat = (3xL) platform displacements from mean position
        bhi = numerator coefficients for high pass filter
        ahi = denominator coefficients for high pass filter
        sf = sampling frequency
        accm = measured platform linear accelerations
        euler = (3XL) euler angles phi, theta, psi
        L = number of measurements

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """
    # keep DPS variable names and comments
    gravxyz = np.mean(accm, axis=-1)
    gravity = np.sqrt(grvxyz.dot(gravxyz))

    # rotate measured accelerations into earth frame
    acc = trans(accm, euler, L, True)
    # remove gravity
    acc[3, :] = acc[3, :] - gravity

    # integrate accelerations to get velocities
    uvwplat = sp.integrate.cumtrapz(acc, axis=-1, initial=0.0) / sf
    # DPS filtfilter is coded as filtfilt with constant padding; padding length is hardcoded at 20.
    uvwplat = sp.signal.filtfilt(bhi, ahi, uvwplat, axis=-1, padtype='constant', padlen=20)

    # integrate again to get displacements
    xyzplat = sp.integrate.cumtrapz(uvwplat, axis=-1, initial=0.0) / sf
    # DPS filtfilter function is coded as filtfilt with constant padding;
    # padding length is hardcoded in the DPS at 20.
    xyzplat = sp.signal.filtfilt(bhi, ahi, xyzplat, axis=-1, padtype='constant', padlen=20)

    return acc, uvwplat, xyzplat


def fdc_fdchp_anglesclimodeyaw(ahi, bhi, sf, accm, ratem, gyro, its, goodcompass, L):
    """
    Description:

        Calculates the euler angles for the FDCHP instrument platform. This code is straight from
        the FDCHP DPS.

    Implemented by:

        2014-05-19: Russell Desiderio. Initial Code

    Usage:

        [euler, dr] = fdc_fdchp_anglesclimodeyaw(ahi, bhi, sf, accm, ratem, gyro, its, goodcompass, L)

            where

        euler = (3xL) array of euler angles (phi, theta, psi) in radians
        dr = corrected angular rate velocities
        ahi = denominator coefficients for high pass filter
        bhi = numerator coefficients for high pass filter
        sf = sampling frequency
        accm = (3xL) array of recalibrated platform linear accelerations
        ratem = (3xL) array of recalibrated angular rates
        gyro = (1xL) array of gyro signal
        its = number of iterations
        goodcompass = boolean signifying whether gyro measurements are to be used.
        L = number of measurements

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """
    # keep DPS variable names and comments
    gravxyz = np.mean(accm, axis=-1)
    gravity = np.sqrt(grvxyz.dot(gravxyz))

    # unwrap compass
    gyro = unwrap(gyro)
    # DPS: "remove mean from rate sensors". DPS code removes linear trend.
    ratem = sp.signal.detrend(ratem, type='linear', axis=-1)

    ### documentation from DPS code verbatim:
    # low frequency angles from accelerometers and gyro
    # slow roll from gravity effects on horizontal accelerations. low pass
    # filter since high frequency horizontal accelerations may be 'real'

    # pitch
    theta = -accm[0, :] / gravity
    mask = np.absolute(theta) < 1
    theta[mask] = np.arcsin(theta[mask])
    theta_slow = theta - sp.signal.filtfilt(bhi, ahi, theta, axis=-1, padtype='constant', padlen=20)

    # roll
    phi = accm[1, :] / gravity
    mask = np.absolute(phi/np.cos(theta_slow)) < 1
    phi[mask] = np.arcsin(phi[mask] / cos(theta_slow(mask)))
    phi_slow = phi - sp.signal.filtfilt(bhi, ahi, phi, axis=-1, padtype='constant', padlen=20)

    ### documentation from DPS code verbatim:
    # yaw
    # here, we estimate the slow heading. the 'fast heading' is not needed
    # for the euler angle update matrix. the negative sign puts the gyro
    # signal into a right handed system.

    fc = 1.0/240.0   # in DPS; does not appear to be used
    ahi2 = [1.0, -3.986978324904284, 5.961019689755254, -3.961104082313157, 0.987062718074823];
    bhi2 = [0.993510300940470, -3.974041203761880, 5.961061805642819, -3.974041203761880, 0.993510300940470];
    if goodcompass:
        psi_slow = -gyro - sp.signal.filtfilt(bhi2, ahi2, gyro, axis=-1, padtype='constant', padlen=20)
    else:
        psi_slow = -np.median(gyro)*np.ones(phi.shape)

    # use slow angles as first guess
    euler = np.vstack((phi_slow, theta_slow, psi_slow))
    rates = update(ratem, euler, L)

    # "i will use this filter with a lower cutoff for yaw"
    # "since the compass is having issues"

    # integrate and filter angle rates, and add to slow angles
    # coded as in DPS, except for goodcompass conditional
    for ii in range(its):
        phi_int = sp.integrate.cumtrapz(rates[0, :], axis=-1, initial=0.0) / sf
        phi = phi_slow + sp.signal.filtfilt(bhi, ahi, phi_int, axis=-1, padtype='constant', padlen=20)
        theta_int = sp.integrate.cumtrapz(rates[1, :], axis=-1, initial=0.0) / sf
        theta = theta_slow + sp.signal.filtfilt(bhi, ahi, theta_int, axis=-1, padtype='constant', padlen=20)
        psi_int = sp.integrate.cumtrapz(rates[2, :], axis=-1, initial=0.0) / sf
        # rad: note that psi_slow values are also a function of the goodcompass value
        if goodcompass:
            psi = psi_slow + sp.signal.filtfilt(bhi2, ahi2, psi_int, axis=-1, padtype='constant', padlen=20)
        else:
            psi = psi_slow + psi_int

        euler = np.vstack((phi, theta, psi))
        rates = update(ratem, euler, L)
        rates = sp.signal.detrend(rates, type='constant', axis=-1)

    dr = ratem

    return euler, dr
