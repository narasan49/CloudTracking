import numpy as np
import scipy.ndimage as nd
from scipy.optimize import curve_fit, minimize

def SearchRangeInDegree(umin, vmin, umax, vmax, lat, dt, deg2grid, Rv):
    """
    [umin, vmin], [umax, vmax]: velocity in unit of [m/s]
    lat: latitude [degree]
    dt: time interval [hr]
    """
    circum = 2*np.pi*Rv*1000. #[m]
    cos_lat = np.cos(lat*np.pi/180.)
    umax2ind = int(umax*dt*3600./(circum*cos_lat)*360.*deg2grid)
    umin2ind = int(umin*dt*3600./(circum*cos_lat)*360.*deg2grid)
    vmax2ind = int(vmax*dt*3600./circum*360.*deg2grid)
    vmin2ind = int(vmin*dt*3600./circum*360.*deg2grid)
    return umin2ind, vmin2ind, umax2ind, vmax2ind

def mps2dph(vel, Rv, lat=None, cos_factor=False):
    """
    convert unit of speed from m s^-1 to degree hr^-1
    """
    circum = 2*np.pi*Rv*1000. #[m]
    if cos_factor:
        cos_lat = np.cos(lat*np.pi/180.)
    else:
        cos_lat = 1.0
    return vel*3600/(circum*cos_lat)*360

def CCSMask(tmpl_shape, cc_shape, targ_mask):
    """
    Return array that masks a corss-correlation surface where the invalid radiance values are used for the calculation.
    input:
        tmpl_shape: tuple of size of template (ny, nx)
        cc_shape: tuple of size of correlation surface (cy, cx)
        targ_mask: mask of search area
    """
    ny, nx = tmpl_shape
    cy, cx = cc_shape
    # print(ny, nx, cy, cx, targ_mask.shape)
    left_bottom  = np.where(targ_mask[ :cy,  :cx]==False, False, True)
    right_bottom = np.where(targ_mask[ :cy, -cx:]==False, False, True)
    left_top     = np.where(targ_mask[-cy:,  :cx]==False, False, True)
    right_top    = np.where(targ_mask[-cy:, -cx:]==False, False, True)
    res = np.logical_and(np.logical_and(left_bottom, right_bottom), np.logical_and(left_top, right_top))
    return res

def FindPeaks2d(data, roi=np.array([]), thresh=0.0, degree=2):
    sm_data = nd.gaussian_filter(data, [degree, degree])
    if roi.ndim == data.ndim:
        """roi を設定した場合、していない場合で共通のピークを選ぶ。"""
        roied_data = sm_data*roi

        roied_xdif = np.roll(roied_data, -1, axis=1) - roied_data
        roied_xdif2 = roied_xdif*np.roll(roied_xdif, 1, axis=1)
        roied_ydif = np.roll(roied_data, -1, axis=0) - roied_data
        roied_ydif2 = roied_ydif*np.roll(roied_ydif, 1, axis=0)

        cand_roied_xdif = np.where(roied_xdif<0, 1, 0)
        cand_roied_xdif2 = np.where(roied_xdif2<0, 1, 0)
        cand_roied_ydif = np.where(roied_ydif<0, 1, 0)
        cand_roied_ydif2 = np.where(roied_ydif2<0, 1, 0)
        cand_roied_thresh = np.where(roied_data>thresh, 1, 0)
        cand_roied = cand_roied_xdif*cand_roied_xdif2*cand_roied_ydif*cand_roied_ydif2*cand_roied_thresh
    else:
        cand_roied = 1

    xdif = np.roll(sm_data, -1, axis=1) - sm_data
    xdif2 = xdif*np.roll(xdif, 1, axis=1)
    ydif = np.roll(sm_data, -1, axis=0) - sm_data
    ydif2 = ydif*np.roll(ydif, 1, axis=0)

    cand_xdif = np.where(xdif<0, 1, 0)
    cand_xdif2 = np.where(xdif2<0, 1, 0)
    cand_ydif = np.where(ydif<0, 1, 0)
    cand_ydif2 = np.where(ydif2<0, 1, 0)
    cand_thresh = np.where(sm_data>thresh, 1, 0)
    ind = np.where(cand_xdif*cand_xdif2*cand_ydif*cand_ydif2*cand_thresh*cand_roied==1)
    return ind

def orientation(img):
    # if not theta0:
    #局所解を避けるため-pi/2~pi/2を10分割してその中から初期値を決める。
    #10個の初期値の評価関数を比較
    def eval_func(theta, img):
        dif = dif_img_along_streak(img, theta)
        var = np.nanvar(dif)
        return var

    theta0_cand = np.linspace(-0.5, 0.5, 10)*np.pi
    phi0_cand   = np.zeros([10])
    for i in range(0,10):
        phi0_cand[i] = eval_func(theta0_cand[i], img)

    theta0_ind = np.argmin(phi0_cand)
    theta0 = theta0_cand[theta0_ind]
    #評価関数の最小化
    theta = minimize(eval_func, theta0, args=(img), method='Nelder-Mead')
    res = theta.x[0]

    #0 < theta < piをとるように
    if res > np.pi:
        res -= np.pi
    if res < 0:
        res += np.pi
    return res

def dif_img_along_streak(img, theta):
    c = np.cos(theta)
    s = np.sin(theta)

    zipj = np.roll(img,-1, axis=1) #z[i+1,j  ]
    zimj = np.roll(img, 1, axis=1) #z[i-1,j  ]
    zijp = np.roll(img,-1, axis=0) #z[i  ,j+1]
    zijm = np.roll(img, 1, axis=0) #z[i  ,j-1]
    res = c*0.5*(zipj-zimj)+s*0.5*(zijp-zijm) # (cos, sin) と grad(z[i,j])の内積
    res[ 0, :] = 0
    res[-1, :] = 0
    res[ :, 0] = 0
    res[ :,-1] = 0
    # res = np.where(res-res == 0, res, 0) #nan埋めを0埋めに変換

    return res

def auto_corr(x):
    n = x.shape[0]
    ave = np.nansum(x)/n
    xp = x-ave
    xp = np.where(xp-xp==0, xp, 0)
    ac  = np.correlate(xp, xp, "full")
    return ac/ac[n-1]

def FitCC(x, y, err, maxind):
    def func_fit(x, a, b, c, d, e):
        # return np.arctanh(a*np.exp(-0.5*((x-b)/c)**2))+d+e*x
        return a*np.exp(-0.5*((x-b)/c)**2)+d+e*x

    initial_guess = [y[maxind], x[maxind], 10, 0, 0]
    n = x.shape[0]
    rng = np.arange(max([0, maxind-30]), min([maxind+30, n]))
    try:
        res = curve_fit(func_fit, x[rng], y[rng], sigma=err[rng], absolute_sigma=True,
                        p0=initial_guess, bounds=([-np.inf, x.min(), 0., -10., -np.inf], [np.inf, x.max(), 100, 10., np.inf]))
        param = res[0]
        perr = np.sqrt(np.diag(res[1]))
    except RuntimeError:
        param = [np.NaN for i in range(5)]
        perr  = [np.NaN for i in range(5)]
    return param[1], perr[1]

def ConvertNdarray2StrList(data, formater):
    if isinstance(data, np.ndarray):
        shape = data.shape
        m = map(lambda x: format(x, formater), data.reshape([-1]))
        data_str = np.array(list(m)).reshape(shape).tolist()

    elif isinstance(data, list):
        data_np = np.array(data)
        shape = data_np.shape
        m = map(lambda x: format(x, formater), data_np.reshape([-1]))
        data_str = np.array(list(m)).reshape(shape).tolist()
    return data_str

def ConvertNdarray2List(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif isinstance(data, list):
        pass
    return data
