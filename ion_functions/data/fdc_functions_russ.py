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
    # constants from the DPS:
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

        u_rot = fdc_fdchp_alignwind(u)

            where

        u_rot = (3xn_rec) wind velocity rotated into the "streamwise" coordinate system
        u = (3Xn_rec) wind velocity in the earth frame (uncorrected for magnetic declination)

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """
    # mean wind velocity components
    u_mean = np.mean(u, axis=-1)

    # calculate angles for coordinate rotation
    u_hor = np.sqrt(u_mean[0] * u_mean[0] + u_mean[1] * u_mean[1])
    beta = np.atan2(u_mean[2], u_hor)
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


def fdc_fdchp_accelsclimode(bhi, ahi, sf, accm, euler, n_rec):
    """
    Description:

        Rotates linear accelerations measured on the FDCHP platform into an earth reference
        system, then integrates to get platform velocity and displacement. This code is
        straight from the FDCHP DPS.

    Implemented by:

        2014-05-19: Russell Desiderio. Initial Code

    Usage:

        acc, uvwplat, xyzplat = fdc_fdchp_accelsclimode(bhi, ahi, sf, accm, euler, n_rec)

            where

        acc = (3xn_rec) linear accelerations in "FLIP/Earth" reference
        uvwplat = (3xn_rec) linear velocities at the point of measurement
        xyzplat = (3xn_rec) platform displacements from mean position
        bhi = numerator coefficients for high pass filter
        ahi = denominator coefficients for high pass filter
        sf = sampling frequency
        accm = measured platform linear accelerations
        euler = (3Xn_rec) euler angles phi, theta, psi
        n_rec = number of measurements

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
    acc = fdc_fdchp_trans(accm, euler, n_rec, True)
    # remove gravity
    acc[2, :] = acc[2, :] - gravity

    # integrate accelerations to get velocities
    uvwplat = sp.integrate.cumtrapz(acc, axis=-1, initial=0.0) / sf
    # DPS filtfilter is coded as filtfilt with constant padding; padding length is hardcoded at 20.
    uvwplat = sp.signal.filtfilt(bhi, ahi, uvwplat, axis=-1, padtype='constant', padlen=20)

    # integrate again to get displacements
    xyzplat = sp.integrate.cumtrapz(uvwplat, axis=-1, initial=0.0) / sf
    xyzplat = sp.signal.filtfilt(bhi, ahi, xyzplat, axis=-1, padtype='constant', padlen=20)

    return acc, uvwplat, xyzplat


def fdc_fdchp_anglesclimodeyaw(ahi, bhi, sf, accm, ratem, gyro, its, goodcompass, n_rec):
    """
    Description:

        Calculates the euler angles for the FDCHP instrument platform. This code is straight from
        the FDCHP DPS.

    Implemented by:

        2014-05-19: Russell Desiderio. Initial Code

    Usage:

        euler, dr = fdc_fdchp_anglesclimodeyaw(ahi, bhi, sf, accm, ratem, gyro, its, goodcompass, n_rec)

            where

        euler = (3xn_rec) array of euler angles (phi, theta, psi) in radians
        dr = corrected angular rate velocities
        ahi = denominator coefficients for high pass filter
        bhi = numerator coefficients for high pass filter
        sf = sampling frequency
        accm = (3xn_rec) array of recalibrated platform linear accelerations
        ratem = (3xn_rec) array of recalibrated angular rates
        gyro = (1xn_rec) array of gyro signal
        its = number of iterations
        goodcompass = boolean signifying whether gyro measurements are to be used.
        n_rec = number of measurements

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
    gyro = np.unwrap(gyro)
    # DPS comment: "remove mean from rate sensors".
    # However, the DPS code removes the linear trend.
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
    ahi2 = [1.0, -3.986978324904284, 5.961019689755254, -3.961104082313157, 0.987062718074823]
    bhi2 = [0.993510300940470, -3.974041203761880, 5.961061805642819, -3.974041203761880, 0.993510300940470]
    if goodcompass:
        psi_slow = -gyro - sp.signal.filtfilt(bhi2, ahi2, gyro, axis=-1, padtype='constant', padlen=20)
    else:
        psi_slow = -np.median(gyro)*np.ones(phi.shape)

    # use slow angles as first guess
    euler = np.vstack((phi_slow, theta_slow, psi_slow))
    rates = fdc_fdchp_update(ratem, euler, n_rec)

    # "i will use this filter with a lower cutoff for yaw"
    # "since the compass is having issues"

    # integrate and filter angle rates, and add to slow angles;
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
        rates = fdc_fdchp_update(ratem, euler, n_rec)
        rates = sp.signal.detrend(rates, type='constant', axis=-1)

    dr = ratem

    return euler, dr


def fdc_fdchp_update(ang_rates, angles, n_rec):
    """
    Description:

        This function is drevied from the EDDYCORR toolbox.
        It computes the angular update matrix as described in
        Edson et al. (1998) and Thwaites (1995) page 50.

    Implemented by:

        2014-05-16: Craig Risien. Initial Code

    Usage:

        values = fdc_fdchp_update(ang_rates, angles, n_rec)

            where

        values = output matrix of values
        ang_rates = (3xN) array of angular rates.
        angles = (3xN) array of Euler angles phi,theta,psi.
        n_rec = number of records. If the FDCHP is sampling at
                10Hz for 20 min, n_rec should equal 12000.

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """

    p = ang_rates[0, 0:n_rec]
    t = ang_rates[1, 0:n_rec]
    ps = ang_rates[2, 0:n_rec]

    up = angles[0, 0:n_rec]
    vp = angles[1, 0:n_rec]
    wp = angles[2, 0:n_rec]

    u = up + vp * np.sin(p) * np.tan(t) + wp * np.cos(p) * np.tan(t)
    v = 0 + vp * np.cos(p) - wp * np.sin(p)
    w = 0 + vp * np.sin(p) / np.cos(t) + wp * np.cos(p) / np.cos(t)

    values = np.vstack((u, v, w))
    return values


def fdc_fdchp_trans(ang_rates, angles, n_rec, iflag=True):
    """
    Description:

        ?

    Implemented by:

        2014-05-16: Craig Risien. Initial Code

    Usage:

        values = fdc_fdchp_trans(ang_rates, angles, n_rec, iflag=True)

            where

        values = output matrix of values
        ang_rates = (3xN) array of angular rates.
        angles = (3xN) array of Euler angles phi,theta,psi.
        iflag = True
        n_rec = number of records. If the FDCHP is sampling at
                10Hz for 20 min, n_rec should equal 12000.

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """

    p = ang_rates[0, 0:n_rec]
    t = ang_rates[1, 0:n_rec]
    ps = ang_rates[2, 0:n_rec]

    up = angles[0, 0:n_rec]
    vp = angles[1, 0:n_rec]
    wp = angles[2, 0:n_rec]

    if iflag:
        u = (up * np.cos(t) * np.cos(ps) + vp * (np.sin(p) * np.sin(t) *
             np.cos(ps) - np.cos(p) * np.sin(ps)) + wp * (np.cos(p) *
             np.sin(t) * np.cos(ps) + np.sin(p) * np.sin(ps)))
        v = (up * np.cos(t) * np.sin(ps) + vp * (np.sin(p) * np.sin(t) *
             np.sin(ps) + np.cos(p) * np.cos(ps)) + wp * (np.cos(p) *
             np.sin(t) * np.sin(ps) - np.sin(p) * np.cos(ps)))
        w = (up * (-np.sin(t)) + vp * (np.cos(t) * np.sin(p)) + wp *
             (np.cos(t) * np.cos(p)))
    else:
        u = (up * np.cos(t) * np.cos(ps) + vp * np.cos(t) * np.sin(ps) -
             wp * np.sin(t))
        v = (up * (np.sin(p) * np.sin(t) * np.cos(ps) - np.cos(p) *
             np.sin(ps)) + vp * (np.sin(p) * np.sin(t) * np.sin(ps) +
             np.cos(p) * np.cos(ps)) + wp * (np.cos(t) * np.sin(p)))
        w = (up * (np.cos(p) * np.sin(t) * np.cos(ps) + np.sin(p) *
             np.sin(ps)) + vp * (np.cos(p) * np.sin(t) * np.sin(ps) -
             np.sin(p) * np.cos(ps)) + wp * (np.cos(t) * np.cos(p)))

    values = np.vstack((u, v, w))
    return values


def fdc_fdchp_sonic(sonics, omegam, euler, uvwplat, dist_vec, n_rec):
    """
    Description:

        This function, which comes from the EDDYCORR toolbox,
        corrects the sonic anemometer components for platform
        motion and orientation.

    Implemented by:

        2014-05-16: Craig Risien. Initial Code

    Usage:

        [uvw, uvwr, uvwrot] = fdc_fdchp_sonic(sonics, omegam, euler, uvwplat, R, n_rec)

            where

        sonics = row of integers corre to sonic numbers which are to be corrected
        omegam = (3xN) measured angular rate 'vector' in platform frame
        euler = (3xN) array of euler angles (phi, theta, psi)
        uvwplat = (3xN) array of platform velocities
        dist_vec = (3x1) distance vector between IMU and Sonic sampling volume
        n_rec = number of records. If the FDCHP is sampling at
                10Hz for 20 min, n_rec should equal 12000.


    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """

    Rvec = np.tile(dist_vec, n_rec)
    uvwrot = np.cross(omegam, Rvec)

    uvwr = fdc_fdchp_trans(sonics + uvwrot, euler, n_rec, True)
    uvw = uvwr + uvwplat


def fdc_fdchp_despikesimple(data):
    """
    Description:

        Function to remove outliers.

    Implemented by:

        2014-05-19: Craig Risien. Initial Code

    Usage:

        [data] = fdc_fdchp_despikesimple(data)

            where

        data = (3xN) array of data values


    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """

    array_size = np.shape(data)
    t = np.arange(0, array_size[1])

    for i in range(0, array_size[0]):
        M = np.median(data[i, :])
        S = np.std(data[i, :])
        ind = np.logical_and(data[i, :] < M + 6 * S, data[i, :] > M - 6 * S)
        f = sp.interpolate.interp1d(t[ind], data[i, ind], kind='nearest')
        data[i, :] = f(t)


def fdc_fdchp_flux(sonicU, sonicV, sonicW, sonicT, heading, roll, pitch,
                   rateX, rateY, rateZ, accX, accY, accZ, lat):
    """
    Description:

        Calculates the L2 data products FLUXMOM-U, FLUXMOM-V, and FLUXHOT from the FDCHP
        instrument. It is anticipated that wrappper functions will be written that will
        call this routine in order to furnish discrete data products. This code is straight
        from the FDCHP DPS.

    Implemented by:

        2014-05-20: Russell Desiderio. Initial Code

    Usage:

        fluxmom_u, fluxmom_v, fluxhot = fdc_fdchp_flux(sonicU, sonicV, sonicW, sonicT,
                                                       heading roll, pitch,
                                                       rateX, rateY, rateZ,
                                                       accX, accY, accZ, lat)

            where

        sonicU, sonicV, sonicW = L0 wind velocities from the sonic anemometer
        sonicT = L0 sonic temperature from the sonic anemometer
        heading, pitch, roll = L0 variables from the magnetometer
        rateX, rateY, rateZ = L0 angular rates
        accX, accY, accZ = L0 linear accelerations
        lat = latitude of instrument in decimal degrees

    References:

        OOI (2014). Data Product Specification for FDCHP Data Products. Document
            Control Number 1341-00280. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00280_Data_Product_Spec_FDCHP_OOI.pdf)
    """
    # hardcoded variables in the DPS code
    n_iter = 5
    goodcompass = False
    roffset = 0.0
    poffset = 0.0
    # z distance between IMU and sonic sampling volume
    z_imu_2_smplvol = 0.85
    # sampling frequency, Hz
    fs = 10.0
    # butterworth filter coefficients
    ahi = [1.0, -3.895833876325378, 5.692892648957244, -3.698121672490411, 0.901065298297355]
    bhi = [0.949244593504399, -3.796978374017597, 5.695467561026395, -3.796978374017597, 0.949244593504399]

    # number of records in one 20 minute deployment;
    # nominally on the order of 10 Hz * 1200 seconds.
    n_rec = sonicU.shape[-1]

    gv = fdc_fdchp_grv(lat)

    # distance vector between IMU and sonic sampling volume
    Rvec = zeros(0.0, 0.0, z_imu_2_smplvol)

    # wind speeds
    sonics = np.vstack(sonicU, sonicV, sonicW)

    # gyro = heading = compass; DPS hardcodes this to the median value.
    gyro = np.radians(heading)

    # in the DPS, 10 points were thrown out on either side of 6000 data points
    # before median value was calculated. so, scale to fs.
    edge = np.around(n_rec / 6000 * fs)
    compass_median = np.median(gyro[edge:-edge])
    # set all gyro values to the median compass value
    gyro = compass_median

    # process angular rate data
    deg_rate = np.radians(np.vstack((rateX, rateY, rateZ)))
    deg_rate = fdc_fdchp_despikesimple(deg_rate)

    # process the linear accelerometer data
    platform = np.vstack((accX, accY, accZ)) * gv
    platform = fdc_fdchp_despikesimple(platform)

    gcomp = np.mean(platform, axis=-1)
    g = np.sqrt(gcomp.dot(gcomp))
    platform = platform * (gv/g)

    platform[0, :] = platform[0, :] + poffset
    platform[1, :] = platform[1, :] + roffset

    gcomp = np.mean(platform, axis=-1)
    g = np.sqrt(gcomp.dot(gcomp))
    platform = platform * (gv/g)

    euler, dr = fdc_fdchp_anglesclimodeyaw(ahi, bhi, fs, platform, deg_rate, gyro, n_iter, goodcompass, n_rec)
    # euler angles are right-handed
    _, uvwplat, _ = fdc_fdchip_accelsclimode(bhi, ahi, fs, platform, euler, n_rec)
    uvw, _, _ = fdc_fdchp_sonic(sonics, dr, euler, uvwplat, Rvec, n_rec)

    # in the DPS, 300 points were thrown out on either side of 6000 data points.
    # scale to fs and total number of points.
    edge = np.around(n_rec / 6000 * fs * 30)
    UVW = uvw[:, edge:-edge]

    # rotate wind velocity components into windstream
    u = alignwind(UVW)
    u = sp.signal.detrend(u, type='linear', axis=-1)

    # set up sonic temperature for buoyancy flux calculation
    Ts = sonicT[edge:-edge]
    u = sp.signal.detrend(Ts, type='linear', axis=-1)

    # calculate flux products
    fluxmom_u = np.mean(u[:, 2] * u[:, 0])
    fluxmom_v = np.mean(u[:, 2] * u[:, 1])
    fluxhot = np.mean(u[:, 2] * Ts)

    return fluxmom_u, fluxmom_v, fluxhot
