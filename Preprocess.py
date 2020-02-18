import numpy as np
import scipy.ndimage as nd
from scipy.linalg import solve
import netCDF4
import datetime

def LoadL3(l3f, varnames):
    res = {}
    nc = netCDF4.Dataset(l3f, "r")
    for varname in varnames:
        value = nc.variables[varname][:]
        unit = nc.variables[varname].getncattr("units")
        if varname in ["radiance", "corrected_radiance"]:
            if unit == "W/m2/sr/m":
                value*=1.0e-9
                unit = "W/m2/sr/nm"
        res[varname] = {"value":value, "unit":unit}
    nc.close()
    return res

def filename2days(filename_without_directory, t_str0):
    #filename = "uvi_20170801_000000_l3b_v20180901.nc"
    t  = datetime.datetime.strptime(filename_without_directory[4:19], '%Y%m%d_%H%M%S')
    t0 = datetime.datetime.strptime(t_str0 , '%Y%m%d_%H%M%S')
    t_flt = (t-t0).total_seconds()/86400.
    return t_flt

def GaussHighPass(img, degree):
    tmp = np.where(img-img==0, img, np.nanmean(img))
    fil = nd.gaussian_filter(tmp, [degree,degree], mode="wrap")
    res = img - fil
    return res

def LinearFit(x, y):
    n     = x.reshape([-1]).shape[0]
    sumx  = np.sum(x)
    sumy  = np.sum(y)
    sumxx = np.sum(x*x)
    sumyy = np.sum(y*y)
    sumxy = np.sum(x*y)

    # Amat x xvec = bvec
    Amat = [[sumxx, sumx],
            [sumx , n   ]]
    bvec =  [sumxy, sumy]

    res = solve(Amat, bvec)
    return res

def MinnaertCorrection(rad, eang, iang, imax=80, emax=80, radmin=1.0e-7):
    ny, nx = rad.shape
    #異常な輝度を除外、入射角・出射角80°以上を除外
    valid_data = np.where((rad > radmin) &
                          (iang< imax) &
                          (eang< emax))

    if valid_data[0].shape[0] > 1000:
        I  = rad[valid_data]
        mu0= np.cos(iang[valid_data]*np.pi/180)
        mu = np.cos(eang[valid_data]*np.pi/180)

        ln1 = np.log(mu*mu0)
        ln2 = np.log(mu*I)

        #ln1, ln2 を線形フィッティング
        res = LinearFit(ln1, ln2)
        crad = rad*np.cos(eang*np.pi/180)/(np.cos(iang*np.pi/180)*np.cos(eang*np.pi/180))**res[0]
        crad = np.where((rad > radmin) &
                        (iang< imax) &
                        (eang< emax), crad, np.NaN)
    else:
        crad=np.array([None])
        res =np.array([None, None])
    return crad, res

def CreateMask(data, vmin=None, vmax=None, nan=True):
    if vmin!=None:
        vmin_mask = np.where(vmin<data, True, False)
    else:
        vmin_mask = np.full_like(data, True)

    if vmax!=None:
        vmax_mask = np.where(data<vmax, True, False)
    else:
        vmax_mask = np.full_like(data, True)

    if nan==True:
        nan_mask = np.where(data-data==0, True, False)
    else:
        nan_mask = np.full_like(data, True)

    return np.logical_and(np.logical_and(vmin_mask, vmax_mask), nan_mask)
