import CTFuncLibrary as fl
import numpy as np
import scipy.ndimage as nd
from scipy.interpolate import interp1d
import cv2

import copy
import json

#Debug
import time
from memory_profiler import profile
import gc
import sys

class CloudTracking():
    class DataForCloudTracking():
        pass

    class CloudTrackingResults():
        pass

    def __init__(
            self, radiances, time, lon, lat, radius,
            tsize, ulimit, vlimit, rad_masks=None, vel_margin=0.0, xdivision=None, ydivision=None,
            eastwest_coordinate="lst", sslon=None,
        ):
        """
        Initialize cloud tracking.
        Input data:
            radiances (nt, ny, nx,): 3-dimensional ndarray.
            rad_masks (nt, ny, nx,): 3-dimensional boolian ndarray.
            time: 1-d ndarray (nt,) of time in an unit of [hours]
            lon: 1-d ndarray (nx,) of longitude in an unit of [degree]
            lat: 1-d ndarray (ny,) of latitude in an unit of [degree]
            sslon: sub-solar longitude.
            tsize: template size of cloud tracking [degree]
            radius: altitude from planetary center in [km]
        """
        if radiances.dtype != np.float32:
            radiances = radiances.astype(np.float32)
        if not isinstance(rad_masks, np.ndarray):
            rad_masks = np.where(radiances-radiances==0, True, False)
        if eastwest_coordinate=="lst":
            if sslon==None:
                raise ValueError("sslon should be set if eastwest_coordinate==\"lst\".")
        self.data = self.DataForCloudTracking()
        self.res = self.CloudTrackingResults()
        deg2grid=int(1/(lon[1]-lon[0]))

        self.deg2grid=deg2grid
        self.tsize=tsize*deg2grid
        self.Rv = radius

        nsp, ny, nx=radiances.shape
        self.nsp=nsp
        self.data.ny=ny
        self.data.nx=nx

        if xdivision==None:
            xdivision = int(nx/self.tsize*2)
        if ydivision==None:
            ydivision = int(ny/self.tsize*2)
        self.xdivision=xdivision
        self.ydivision=ydivision

        sslon_idx = np.argmin(abs(lon-sslon))
        lst = -(np.arange(nx)+0.5)/nx*24.0 + 24.0
        if eastwest_coordinate=="lst":
            radiances = np.roll(radiances, -(sslon_idx-nx//2), axis=2)
            rad_masks = np.roll(rad_masks, -(sslon_idx-nx//2), axis=2)
            lon = np.roll(lon, -(sslon_idx-nx//2), axis=0)
            sslon_idx = nx//2
        elif eastwest_coordinate=="lon":
            lst = np.roll(lst, sslon_idx-nx//2, axis=0)
        else:
            raise ValueError()

        self.data.sslon=sslon
        self.data.ssp=sslon_idx
        """    center longitude and latitude of each template    """
        #indices
        lon_vec_ind = np.array([self.tsize/2*(mx+1) for mx in range(xdivision-1)], dtype=np.int32)
        lat_vec_ind = np.array([self.tsize/2*(my+1) for my in range(ydivision-1)], dtype=np.int32)
        #In degrees
        lon_vec = lon[lon_vec_ind] - 1/deg2grid/2
        lat_vec = lat[lat_vec_ind] - 1/deg2grid/2
        lst_vec = lst[lon_vec_ind] - (lst[1]-lst[0])/2

        self.data.radiances=radiances
        self.data.rad_masks=rad_masks
        self.data.times=time
        self.data.lon=lon
        self.data.lat=lat
        self.data.lst=lst

        self.data.lon_vec_ind=lon_vec_ind
        self.data.lat_vec_ind=lat_vec_ind
        self.data.lon_vec=lon_vec
        self.data.lat_vec=lat_vec
        self.data.lst_vec=lst_vec

        #ulimit: list of [[lat, umin, vmax], ...]
        if np.array(ulimit).shape[1]==3:
            """"""
            umin, umax = self.lerp_limit(ulimit)
            self.umin = umin
            self.umax = umax
        else:
            raise ValueError("array size of ulimit should be list of [[lat, umin, umax], ...]")

        if np.array(vlimit).shape[1]==3:
            """"""
            vmin, vmax = self.lerp_limit(vlimit)
            self.vmin = vmin
            self.vmax = vmax
        else:
            raise ValueError("array size of vlimit should be list of [[lat, vmin, vmax], ...]")
        self.vel_margin = vel_margin
        self.RegionOfInterest()

    def lerp_limit(self, limit):
        lat_vec = self.data.lat_vec
        ny = lat_vec.shape[0]
        limit = np.array(limit)
        min_val = interp1d(limit[:, 0], limit[:, 1])(lat_vec)
        max_val = interp1d(limit[:, 0], limit[:, 2])(lat_vec)
        return min_val, max_val


    def CrossCorrelation(self, dif_streak=True, cc_mask_type="fill_zero"):
        """
        Conduct clculation of cross-correlation.
        parameters:
            dif_streak: If an elimination of streaks is applied (True) or not (False). The deafault is True.
            cc_mask_type: 
        results:
            self.res.cc: cross-correlation
            self.res.xind, self.res.yind: indices of sub-images used to cross-correlation calculation
            self.res.theta: determined orientation which is dominant in each subimage
        """
        if cc_mask_type=="fill_zero":
            mask_value = 0.0
        elif cc_mask_type=="ignore":
            mask_value = np.NaN
        elif cc_mask_type=="no_mask":
            pass
        else:
            raise ValueError("Invalid cc_mask_type! cc_mask_type should be fill_zero, ignore, or no_mask.")
        r    = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp-1)]
        xind = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp)]
        yind = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp)]
        theta=  [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
        cc_uaxis = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp-1)]
        cc_vaxis = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp-1)]

        for mx in range(self.xdivision-1):
            for my in range(self.ydivision-1):
                minlon = int(self.tsize/2*(mx+1))
                minlat = int(self.tsize/2*(my+1))
                rad=[np.NaN]*self.nsp
                rad_mask = [np.NaN]*self.nsp
                """Cut out sub-images"""
                if_calc_cc=True
                for ti in range(self.nsp):
                # for ti in range(self.nsp-1, -1, -1):
                    dt = self.data.times[ti]-self.data.times[0]
                    uminIdx, vminIdx, umaxIdx, vmaxIdx = fl.SearchRangeInDegree(
                        self.umin[my]-self.vel_margin, self.vmin[my]-self.vel_margin,
                        self.umax[my]+self.vel_margin, self.vmax[my]+self.vel_margin,
                        self.data.lat_vec[my], dt, self.deg2grid, self.Rv,
                    )
                    yindi = np.arange(minlat+vminIdx, minlat+self.tsize+vmaxIdx) # Y-indices of sub images
                    xindi = np.arange(minlon+uminIdx, minlon+self.tsize+umaxIdx) # X-indices of sub images

                    if (yindi<0).any() or (self.data.ny<=yindi).any():
                        if_calc_cc=False
                        break
                    xindi = np.where(xindi >= self.data.nx, xindi-self.data.nx, xindi)
                    xindi = np.where(xindi < 0, xindi+self.data.nx, xindi)

                    radi = self.data.radiances[ti][yindi][:,xindi]
                    radi_mask = self.data.rad_masks[ti][yindi][:,xindi]

                    # if ti==0:
                    if not (radi_mask.all()):
                        if_calc_cc=False
                        break
                    
                    # deg_lowpass = (.5+1*abs(self.data.lat_vec[my])/90.0)*5
                    # deg_lowpass = 8
                    # radi = nd.gaussian_filter(radi, [deg_lowpass,deg_lowpass]) #low-pass filter
                    if dif_streak:
                        # if ti==self.nsp-1:
                        if ti==0:
                            theta[my][mx] = fl.orientation(radi) # determine orientation of streak
                        radi = fl.dif_img_along_streak(radi, theta[my][mx]) #differentiate
                    else:
                        theta[my][mx] = None
                    if ti >=1:
                        u_min_deg = fl.mps2dph(self.umin[my]-self.vel_margin, self.Rv, lat=self.data.lat_vec[my], cos_factor=True)
                        v_min_deg = fl.mps2dph(self.vmin[my]-self.vel_margin, self.Rv, cos_factor=False)
                        cc_uaxis[ti-1][my][mx] = np.arange(uminIdx, umaxIdx+1)/self.deg2grid/dt #+ u_min_deg
                        cc_vaxis[ti-1][my][mx] = np.arange(vminIdx, vmaxIdx+1)/self.deg2grid/dt #+ v_min_deg
                        # print(cc_uaxis[ti-1][my][mx][0], cc_uaxis[ti-1][my][mx][-1], u_min_deg)
                    xind[ti][my][mx] = xindi.tolist()
                    yind[ti][my][mx] = yindi.tolist()
                    rad[ti] = radi
                    rad_mask[ti] = radi_mask
                if if_calc_cc:
                    for i in range(self.nsp-1):
                        rad_tmp = np.where(rad[i+1]-rad[i+1]==0, rad[i+1], 0.0).astype(np.float32)
                        cc = cv2.matchTemplate(rad_tmp, rad[0], cv2.TM_CCOEFF_NORMED) # calculate cross-correlation surface
                        if cc_mask_type!="no_mask":
                            # print(rad_mask[i+1])
                            cc_mask = fl.CCSMask(rad[0].shape, cc.shape, rad_mask[i+1])
                            cc = np.where(cc_mask==True, cc, mask_value).astype(np.float32)
                        r[i][my][mx] = cc
        self.res.theta = theta
        self.res.cc = r
        self.res.xind = xind
        self.res.yind = yind
        self.res.cc_uaxis = cc_uaxis
        self.res.cc_vaxis = cc_vaxis
        gc.collect()
    # @profile
    def SuperPosition(self, spatial=True):
        """
        Superpose the CCSs obtained with diferent pair.
        The region where obtained CCSs were 0 or 1 was ignored (fill blank ndarray).
        keyward:
            spatial: if it set to be True, cross-correlation surfaces are superposed spatially as well as temporally.

        results:
            self.res.cc_sp: list of superposed cross-correlation surfaces
            self.res.num_sp: list of the number of superposed correlation surfaces in each grid
        """

        selec = [[[1 if np.array(cc).ndim==2 else 0 for cc in cc_ti_row] for cc_ti_row in cc_ti] for cc_ti in self.res.cc]
        selec = np.where(np.array(selec).sum(axis=0)==(self.nsp-1))
        selec = np.array([selec[0], selec[1]]).transpose().tolist()
        self.res.selec = selec
        res_cc_sp = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
        res_n_sp  = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
        for my, mx in selec:
            cc_tsp  = np.array(self.res.cc[-1][my][mx], dtype=np.float32)
            cc_tsp  = np.where(cc_tsp-cc_tsp==0, cc_tsp, 0)
            num_tsp = np.where(cc_tsp-cc_tsp==0, 1, 0).astype(np.int32)
            ysize_tsp, xsize_tsp = cc_tsp.shape
            for ti in range(0,self.nsp-2):
                cc = np.array(self.res.cc[ti][my][mx], dtype=np.float32)
                cc_resize = cv2.resize(cc, (xsize_tsp, ysize_tsp))

                #superpose
                cc_tsp = np.where(cc_resize-cc_resize==0, cc_tsp+cc_resize, cc_tsp)
                num_tsp = np.where(cc_resize-cc_resize==0, num_tsp+1, num_tsp)
            if not spatial:
                cc_tsp /= num_tsp

            cc_tsp = np.where(num_tsp==0, np.NaN, cc_tsp)
            res_cc_sp[my][mx] = cc_tsp.astype(np.float32)#.tolist()
            res_n_sp[my][mx] = num_tsp
        # gc.collect()

        if spatial:
            res_cc_stsp  = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
            res_n_stsp   = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
            res_uax_stsp = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
            res_vax_stsp = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]
            for my, mx in selec:
                xneibs = [[my, mx-1], [my, mx+1]]
                yneibs = [[my-1, mx], [my+1, mx]]
                num_xneib = np.sum([1 if xneib in selec else 0 for xneib in xneibs])
                num_yneib = np.sum([1 if yneib in selec else 0 for yneib in yneibs])
                neibs = []
                if num_xneib==2: neibs.extend(xneibs)
                if num_yneib==2: neibs.extend(yneibs)
                
                cc0 = np.array(res_cc_sp[my][mx], dtype=np.float32)
                num_sp0 = np.array(res_n_sp[my][mx], dtype=np.int32)
                ny_cc0, nx_cc0 = cc0.shape
                c0_uax = self.res.cc_uaxis[-1][my][mx]
                c0_vax = self.res.cc_vaxis[-1][my][mx]
                for neib in neibs:
                    cc1 = np.array(res_cc_sp[neib[0]][neib[1]], dtype=np.float32)
                    num_sp1 = np.array(res_n_sp[neib[0]][neib[1]], dtype=np.int32)
                    ny_cc1, nx_cc1 = cc1.shape
                    c1_uax = self.res.cc_uaxis[-1][neib[0]][neib[1]]
                    c1_vax = self.res.cc_vaxis[-1][neib[0]][neib[1]]

                    du = c0_uax[1]-c0_uax[0]
                    # print(int(round((max([c0_uax[-1], c1_uax[-1]]) - min([c0_uax[0], c1_uax[0]]))/du)))
                    nx_new = int(round((max([c0_uax[-1], c1_uax[-1]]) - min([c0_uax[0], c1_uax[0]]))/du)) + 1
                    ny_new = int(round((max([c0_vax[-1], c1_vax[-1]]) - min([c0_vax[0], c1_vax[0]]))/du)) + 1
                    uax_new = np.linspace(min([c0_uax[0], c1_uax[0]]), max([c0_uax[-1], c1_uax[-1]]), nx_new)
                    vax_new = np.linspace(min([c0_vax[0], c1_vax[0]]), max([c0_vax[-1], c1_vax[-1]]), ny_new)
                    # print(uax_new.shape, vax_new.shape)

                    cc0_ = np.zeros([ny_new, nx_new])
                    cc1_ = np.zeros([ny_new, nx_new])
                    num_sp0_ = np.zeros([ny_new, nx_new])
                    num_sp1_ = np.zeros([ny_new, nx_new])
                    # print(np.where(uax_new==c0_uax[0]), c0_uax[0], np.argmin(abs(uax_new-c0_uax[0])))
                    uMinIdx_cc0 = np.argmin(abs(uax_new-c0_uax[0]))
                    vMinIdx_cc0 = np.argmin(abs(vax_new-c0_vax[0]))
                    uMinIdx_cc1 = np.argmin(abs(uax_new-c1_uax[0]))
                    vMinIdx_cc1 = np.argmin(abs(vax_new-c1_vax[0]))
                    # print(uMinIdx_cc0)
                    cc0_[vMinIdx_cc0:vMinIdx_cc0+ny_cc0, uMinIdx_cc0:uMinIdx_cc0+nx_cc0] = cc0
                    cc1_[vMinIdx_cc1:vMinIdx_cc1+ny_cc1, uMinIdx_cc1:uMinIdx_cc1+nx_cc1] = cc1
                    num_sp0_[vMinIdx_cc0:vMinIdx_cc0+ny_cc0, uMinIdx_cc0:uMinIdx_cc0+nx_cc0] = num_sp0
                    num_sp1_[vMinIdx_cc1:vMinIdx_cc1+ny_cc1, uMinIdx_cc1:uMinIdx_cc1+nx_cc1] = num_sp1

                    cc_sum = cc0_ + cc1_
                    num_sp_sum = num_sp0_ + num_sp1_

                    #Update parameters
                    cc0, num_sp0 = cc_sum, num_sp_sum
                    c0_uax = uax_new
                    c0_vax = vax_new
                    ny_cc0, nx_cc0 = ny_new, nx_new
                cc0/=num_sp0
                res_cc_stsp[my][mx] = cc0
                res_n_stsp[my][mx] = num_sp0
                res_uax_stsp[my][mx] = c0_uax
                res_vax_stsp[my][mx] = c0_vax

                # cc_stsp = np.array(res_cc_sp[my][mx], dtype=np.float32)
                # cc_xax = self.res.cc_xaxis[-1][my][mx]
                # cc_yax = self.res.cc_yaxis[-1][my][mx]
                # cc_stsp = np.where(cc_stsp-cc_stsp==0, cc_stsp, 0)
                # n_stsp = np.array(res_n_sp[my][mx], dtype=np.int32)
                # cent_ny, cent_nx = cc_stsp.shape
                # for neib in neibs:
                #     cc_neib = np.array(res_cc_sp[neib[0]][neib[1]], dtype=np.float32)
                #     n_neib = np.array(res_n_sp[neib[0]][neib[1]], dtype=np.int32)
                #     neib_ny, neib_nx = cc_neib.shape

                #     x_ind_min_1 = max([0, cent_nx-neib_nx])
                #     x_ind_min_2 = max([0, neib_nx-cent_nx])
                #     # if cent_ny > neib_ny:
                #     #     ymin_neib = (cent_ny-neib_ny)//2
                #     #     ymax_neib = (cent_ny+neib_ny)//2
                #     #     ymin_cent = 0
                #     #     ymax_cent = cent_ny
                #     # else:
                #     #     ymin_neib = 0
                #     #     ymax_neib = neib_ny
                #     #     ymin_cent = (neib_ny-cent_ny)//2
                #     #     ymax_cent = (neib_ny+cent_ny)//2
                    
                    
                #     cc_neib_cut = cc_neib[:, x_ind_min_2:neib_nx]
                #     n_neib_cut = n_neib[:, x_ind_min_2:neib_nx]
                #     cc_stsp_cut = cc_stsp[:, x_ind_min_1:cent_nx]
                #     n_stsp_cut = n_stsp[:, x_ind_min_1:cent_nx]
                #     cc_stsp[:, x_ind_min_1:cent_nx] += cc_neib[:, x_ind_min_2:neib_nx]
                #     cc_stsp[:, x_ind_min_1:cent_nx] = np.where(cc_neib_cut-cc_neib_cut==0, cc_stsp_cut+cc_neib_cut, cc_stsp_cut)
                #     n_stsp[:, x_ind_min_1:cent_nx] = np.where(cc_neib_cut-cc_neib_cut==0, n_stsp_cut+n_neib_cut, n_stsp_cut)

                # cc_stsp/=n_stsp
                # res_cc_stsp[my][mx] = cc_stsp
                # res_n_stsp[my][mx] = n_stsp

            self.res.cc_sp=res_cc_stsp
            self.res.num_sp=res_n_stsp
            self.res.uax_sp = res_uax_stsp
            self.res.vax_sp = res_vax_stsp

            del res_cc_sp, res_n_sp, res_cc_stsp, res_n_stsp
            gc.collect()
        else:
            self.res.cc_sp=res_cc_sp
            self.res.num_sp=res_n_sp
            self.res.uax_sp = self.res.cc_uaxis[-1]
            self.res.vax_sp = self.res.cc_vaxis[-1]
            del res_cc_sp, res_n_sp
            gc.collect()

    def RegionOfInterest(self):
        roi = []
        dt = self.data.times[-1]-self.data.times[0]
        for i in range(self.ydivision-1):
            uminIdx0, vminIdx0, umaxIdx0, vmaxIdx0 = fl.SearchRangeInDegree(
                self.umin[i]-self.vel_margin,
                self.vmin[i]-self.vel_margin,
                self.umax[i]+self.vel_margin,
                self.vmax[i]+self.vel_margin,
                self.data.lat_vec[i],
                dt,
                self.deg2grid,
                self.Rv
            )
            uminIdx, vminIdx, umaxIdx, vmaxIdx = fl.SearchRangeInDegree(self.umin[i], self.vmin[i], self.umax[i], self.vmax[i], self.data.lat_vec[i], dt, self.deg2grid, self.Rv)
            roi_y = np.zeros([vmaxIdx0-vminIdx0+1, umaxIdx0-uminIdx0+1])

            maxROIx = umaxIdx - uminIdx0
            minROIx = uminIdx - uminIdx0
            maxROIy = vmaxIdx - vminIdx0
            minROIy = vminIdx - vminIdx0
            roi_y[minROIy:maxROIy, minROIx:maxROIx]=1
            roi.append(roi_y)
        self.roi = roi


    def Relaxation(self, a=0.2, d=1.0, thresh=0.2):
        """
        Selection of reliable peak in cross-correlation surface using relaxation labeling method. The details are given by Kouyama et al. (2012)
        """
        xdivision = self.xdivision
        ydivision = self.ydivision
        selec = self.res.selec

        candidates = [[(np.array([]), np.array([])) for i in range(xdivision-1)] for j in range(ydivision-1)]
        p = [[np.array([]) for i in range(xdivision-1)] for j in range(ydivision-1)]
        for my, mx in selec:
            cc = np.array(self.res.cc_sp[my][mx])
            if cc.ndim==2:
                uax = self.res.uax_sp[my][mx]
                vax = self.res.vax_sp[my][mx]
                ny, nx = cc.shape
                roi = np.zeros_like(cc)
                dt = self.data.times[-1]-self.data.times[0]
                u_min_deg = fl.mps2dph(self.umin[my], self.Rv, lat=self.data.lat_vec[my], cos_factor=True)
                u_max_deg = fl.mps2dph(self.umax[my], self.Rv, lat=self.data.lat_vec[my], cos_factor=True)
                v_min_deg = fl.mps2dph(self.vmin[my], self.Rv, cos_factor=False)
                v_max_deg = fl.mps2dph(self.vmax[my], self.Rv, cos_factor=False)
                u_roi = np.where(u_min_deg<=uax, 1, 0)*np.where(uax<=u_max_deg, 1, 0)
                v_roi = np.where(v_min_deg<=vax, 1, 0)*np.where(vax<=v_max_deg, 1, 0)
                u_roi_mat, v_roi_mat = np.meshgrid(u_roi, v_roi)
                roi = u_roi_mat * v_roi_mat
                # u_roi_idx = np.where(u_roi==1)[0]
                # v_roi_idx = np.where(v_roi==1)[0]
                
                # print(roi.shape, u_roi_idx[0].shape, v_roi_idx[0].shape)
                # roi[v_roi_idx][:, u_roi_idx] = np.ones([u_roi_idx.size, v_roi_idx.size])
                # roi[v_roi_idx][:, u_roi_idx] = 1
                # roi[30,300] = 1
                # print(roi.sum(), roi[30, 300], roi.shape, u_roi.sum(), v_roi.sum(), u_min_deg, u_max_deg, uax[0], uax[-1])
                
                candidates[my][mx] = fl.FindPeaks2d(cc, roi=roi, thresh=thresh)
                p[my][mx] = cc[candidates[my][mx]]
            else:
                candidates[my][mx] = (np.array([]), np.array([]))
                p[my][mx] = np.array([])

        p_pre = copy.deepcopy(p)
        opt_u = np.full([ydivision-1, xdivision-1], np.NaN)
        opt_v = np.full([ydivision-1, xdivision-1], np.NaN)
        for my, mx in selec:
            peaky, peakx = candidates[my][mx]
            if peaky.shape[0] >= 1: #if not brank...
                Ik = peakx.shape[0]
                q = np.zeros([Ik])
                for i in range(Ik):
                    #Gk: neighboring templates
                    Gk = [[my-1, mx], [my+1, mx], [my, mx-1], [my, mx+1]]
                    for neib in Gk:
                        if neib in selec:
                            myp, mxp = neib
                            peakyp, peakxp = candidates[myp][mxp]
                            Ikp = peakxp.shape[0]
                            for ip in range(Ikp):
                                q[i] += np.exp(-a/d*((peakx[i]-peakxp[ip])**2+(peaky[i]-peakyp[ip])**2))*p_pre[myp][mxp][ip]

                sum_pq = np.sum(p_pre[my][mx]*q)
                for i in range(Ik):
                    p[my][mx][i] = p_pre[my][mx][i]*q[i]/sum_pq

                maxp_ind = np.argmax(p[my][mx])
                opt_u[my][mx] = peakx[maxp_ind]
                opt_v[my][mx] = peaky[maxp_ind]

        self.res.u_grid=opt_u
        self.res.v_grid=opt_v

    def SubPixelEstimation(self):
        """
        sub-pixel estimation of the peak position obtained by relaxation labeling
        with error estimation
        resultant velocities and errors are stored in 
            self.res.u_sub_grid,
            self.res.v_sub_grid,
            self.res.u_err_grid,
            self.res.v_err_grid
        """

        cc_err = np.full([self.ydivision-1, self.xdivision-1], np.NaN)
        usub = np.full([self.ydivision-1, self.xdivision-1], np.NaN)
        vsub = np.full([self.ydivision-1, self.xdivision-1], np.NaN)
        uerr = np.full([self.ydivision-1, self.xdivision-1], np.NaN)
        verr = np.full([self.ydivision-1, self.xdivision-1], np.NaN)

        u_opt = np.array(self.res.u_grid)
        v_opt = np.array(self.res.v_grid)
        u_valid = np.where(u_opt-u_opt==0, 1, 0)
        v_valid = np.where(v_opt-v_opt==0, 1, 0)
        selec = np.where(u_valid*v_valid==1)
        selec = np.array([selec[0], selec[1]]).transpose().tolist()

        for my, mx in selec:
            selec_ssp = [[my, mx], [my-1, mx], [my+1, mx], [my, mx-1], [my, mx+1]]
            nssp = 0
            sq_err = 0
            u_grid0=int(self.res.u_grid[my][mx])
            v_grid0=int(self.res.v_grid[my][mx])
            for neib in selec_ssp:
                if neib in selec:
                    yi, xi = neib
                    
                    nssp+=1
                    u_grid=self.res.u_grid[yi][xi]
                    v_grid=self.res.v_grid[yi][xi]

                    uinds, vinds, imgs2 = [], [], []
                    img1 = self.data.radiances[0][self.res.yind[0][yi][xi]][:,self.res.xind[0][yi][xi]]
                    theta = self.res.theta[yi][xi]
                    if theta: img1 = fl.dif_img_along_streak(img1, theta)
                    ny1, nx1 = img1.shape

                    cc_n = np.array(self.res.cc[-1][yi][xi])
                    for i in range(self.nsp-1):
                        cc = np.array(self.res.cc[i][yi][xi])
                        uMinIdx = int(cc.shape[1]/cc_n.shape[1]*u_grid)
                        vMinIdx = int(cc.shape[0]/cc_n.shape[0]*v_grid)
                        urange = np.arange(uMinIdx, uMinIdx+nx1)
                        vrange = np.arange(vMinIdx, vMinIdx+ny1)
                        urange = np.where(urange<self.data.nx, urange, urange-self.data.nx)
                        uinds.append(urange)
                        vinds.append(urange)
                        
                        img2xMin = self.res.xind[i+1][yi][xi][0]
                        img2yMin = self.res.yind[i+1][yi][xi][0]
                        img2 = self.data.radiances[i+1][vrange+img2yMin][:, urange+img2xMin]
                        if theta: img2 = fl.dif_img_along_streak(img2, theta)
                        imgs2.append(img2)

                    """
                    calculate degree of freedom near the peak position
                    ac1, ac2: auto-correlation of img1, img2
                    omega: correlation length
                    dof: effective degree of degree of freedom
                    """
                    n = ny1*nx1
                    ac1 = fl.auto_corr(img1.reshape([-1]))
                    for img2 in imgs2:
                        ny2, nx2 = img2.shape
                        # ac2 = fl.auto_corr(img2[vind:vind+ny1][:,uind:uind+nx1].reshape([-1]))
                        # ac2 = fl.auto_corr(img2[vind][:,uind].reshape([-1]))
                        ac2 = fl.auto_corr(img2.reshape([-1]))
                        tau = np.arange(-n+1,n)
                        omega = np.nansum((1-abs(tau)/n)*ac1*ac2)
                        dof = n/omega
                        err_ti = 1.0/np.sqrt(dof-3)
                        sq_err += err_ti**2

            n_sp = self.res.num_sp[my][mx][v_grid0][u_grid0]
            err = np.sqrt(sq_err/n_sp)
            cc_err[my, mx]=err
            cc_tsp = np.array(self.res.cc_sp[my][mx])
            cc_tsp = np.where(cc_tsp-cc_tsp==0, cc_tsp, 0.0)

            if np.nanmax(cc_tsp)-err >0:
                """sub-pixel estimation with non-linear least square fitting"""
                nccy, nccx = cc_tsp.shape
                cc_err_x = np.full(nccx, err)
                cc_err_y = np.full(nccy, err)
                
                # import matplotlib.pyplot as plt
                # plt.imshow(cc_tsp)
                # plt.plot(u_grid0, v_grid0, ".")
                # plt.show()
                usub[my, mx], uerr[my, mx] = fl.FitCC(np.arange(nccx), cc_tsp[v_grid0], cc_err_x, u_grid0)
                vsub[my, mx], verr[my, mx] = fl.FitCC(np.arange(nccy), cc_tsp[:,u_grid0], cc_err_y, v_grid0)
                # print(my, uerr[my, mx], verr[my, mx])
        self.res.u_sub_grid = usub
        self.res.u_err_grid = uerr
        self.res.v_sub_grid = vsub
        self.res.v_err_grid = verr
        self.res.cc_err = cc_err
    def ConvertVectorUnit(self):
        """
        convert unit from [grid hr^-1] to [m s^-1]
        convert target: self.res.u_sub_grid, self.res.v_sub_grid, self.res.u_err_grid, self.res.v_err_grid
        convert result: self.res.u_sub_meter, self.res.v_sub_meter, self.res.u_err_meter, self.res.v_err_meter
        """
        dt = self.data.times[-1]-self.data.times[0]
        circum = 2*np.pi*self.Rv #[km]
        cos_lat = np.cos(self.data.lat_vec*np.pi/180)
        cos_lat_mat = np.dot(cos_lat.reshape([-1, 1]), np.ones([self.xdivision-1]).reshape([1, -1]))
        umin_mat = np.dot(np.array(self.umin).reshape([-1, 1]), np.ones([self.xdivision-1]).reshape([1, -1]))
        vmin_mat = np.dot(np.array(self.vmin).reshape([-1, 1]), np.ones([self.xdivision-1]).reshape([1, -1]))

        u     = self.res.u_sub_grid/self.deg2grid*circum*np.array(cos_lat_mat)/360/dt/3.6 + umin_mat - self.vel_margin
        v     = self.res.v_sub_grid/self.deg2grid*circum/360/dt/3.6 + vmin_mat - self.vel_margin
        u_err = self.res.u_err_grid/self.deg2grid*circum*np.array(cos_lat_mat)/360/dt/3.6
        v_err = self.res.v_err_grid/self.deg2grid*circum/360/dt/3.6

        self.res.u_sub_meter = u
        self.res.v_sub_meter = v
        self.res.u_err_meter = u_err
        self.res.v_err_meter = v_err

    def OutputResults(self, file, ccfile=None):
        output_data = {}
        output_data["u"]    = {"unit": "m s^-1", "value": fl.ConvertNdarray2List(self.res.u_sub_meter)}
        output_data["v"]    = {"unit": "m s^-1", "value": fl.ConvertNdarray2List(self.res.v_sub_meter)}
        output_data["u_err"]= {"unit": "m s^-1", "value": fl.ConvertNdarray2List(self.res.u_err_meter)}
        output_data["v_err"]= {"unit": "m s^-1", "value": fl.ConvertNdarray2List(self.res.v_err_meter)}
        output_data["t"]    = {"unit": "hours since 2000-01-01T00:00:00", "value": fl.ConvertNdarray2List(self.data.times)}
        output_data["lon"]  = {"unit": "degree", "value": fl.ConvertNdarray2List(self.data.lon_vec)}
        output_data["lat"]  = {"unit": "degree", "value": fl.ConvertNdarray2List(self.data.lat_vec)}
        output_data["lst"]  = {"unit": "hours" , "value": fl.ConvertNdarray2List(self.data.lst_vec)}

        with open(file,'w') as f:
            json.dump(output_data,f,indent=4, ensure_ascii=False)

        if ccfile != None:
            output_data = {}
            output_data["cc"]    = {"unit": "", "value": fl.ConvertNdarray2StrList(ct.res.cc_sp, "+.6e")}
