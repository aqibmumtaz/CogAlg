'''
Comp_slice is a terminal fork of intra_blob.
-
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat or high-M blobs.
(high match: M / Ma, roughly corresponds to low gradient: G / Ga)
-
Vectorization is clustering of Ps + their derivatives (derPs) into PPs: patterns of Ps that describe an edge.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (2D alg, 3D alg), this dimensionality reduction is done in salient high-aspect blobs
(likely edges / contours in 2D or surfaces in 3D) to form more compressed skeletal representations of full-D patterns.
-
Please see diagram:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/comp_slice_flip.drawio
'''

from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
from slice_utils import draw_PP_

import warnings  # to detect overflow issue, in case of infinity loop
warnings.filterwarnings('error')

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = .1
flip_ave_FPP = 5  # flip large FPPs only
div_ave = 200
ave_dX = 10  # difference between median x coords of consecutive Ps
ave_Dx = 10
ave_mP = 20  # just a random number right now.
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20

class CDert(ClusterStructure):
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    Mdx = int
    Ddx = int
    flip_val = int

class CP(ClusterStructure):

    Dert = object  # summed kernel parameters
    L = int
    x0 = int
    dX = int  # shift of average x between P and _P, if any
    y = int  # for visualization only
    sign = NoneType  # sign of gradient deviation
    dert_ = list   # array of pixel-level derts: (p, dy, dx, g, m), extended in intra_blob
    upconnect_ = list
    downconnect_cnt = int
    derP = object # derP object reference
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list

class CderP(ClusterStructure):
    ## derDert
    mP = int
    dP = int
    mx = int
    dx = int
    mL = int
    dL = int
    mDx = int
    dDx = int
    mDy = int
    dDy = int
    P = object    # lower comparand
    _P = object   # higher comparand
    PP = object   # FPP if flip_val, contains this derP
    # not needed?:
    fxflip = bool  # flag: splicing | flipping point
    # from comp_dx
    fdx = NoneType
    # optional:
    dDdx = int
    mDdx = int
    dMdx = int
    mMdx = int

class CPP(ClusterStructure):

    Dert  = object  # set of P params accumulated in PP
    derPP = object  # set of derP params accumulated in PP
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType
    box = list   # for visualization only, original box before flipping
    xflip_derP_ = list    # derPs at potential splicing points
    xflip_derP_PP_ = list   # potentially spliced PPs in FPP
    # FPP params
    flip_val = int  # vertical bias in Ps
    dert__ = list
    mask__ = bool
    # PP params
    derP__ = list
    P__ = list
    # PP FPP params
    derPf__ = list
    Pf__ = list
    PPmm_ = list
    PPdm_ = list
    PPmmf_ = list
    PPdmf_ = list
    # PPd params
    derPd__ = list
    Pd__ = list
    # PPd FPP params
    derPdf__ = list
    Pdf__ = list
    PPmd_ = list
    PPdd_ = list
    PPmdf_ = list
    PPddf_ = list    # comp_dx params

# Functions:
'''
leading '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
trailing '_' denotes array name, vs. same-name elements of that array. '__' is a 2D array
leading 'f' denotes flag
-
rough workflow:
-
intra_blob -> slice_blob(blob) -> derP_ -> PP,
if flip_val(PP is FPP): pack FPP in blob.PP_ -> flip FPP.dert__ -> slice_blob(FPP) -> pack PP in FPP.PP_
else       (PP is PP):  pack PP in blob.PP_
'''

def slice_blob(blob, verbose=False):
    '''
    Slice_blob converts selected smooth-edge blobs (high G, low Ga) into sliced blobs,
    adding horizontal blob slices: Ps or 1D patterns
    '''
    if not isinstance(blob, CPP):  # input is blob, else FPP, no flipping
        flip_eval_blob(blob)

    dert__ = blob.dert__
    mask__ = blob.mask__
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")

    for fPPd in range(2):  # run twice, 1st loop fPPd=0: form PPs, 2nd loop fPPd=1: form PPds

        P__ , derP__, Pd__, derPd__ = [], [], [], []
        zip_dert__ = zip(*dert__)
        _P_ = form_P_(list(zip(*next(zip_dert__))), mask__[0], 0)  # 1st upper row
        P__ += _P_  # frame of Ps

        for y, dert_ in enumerate(zip_dert__, start=1):  # scan top down
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            P_ = form_P_(list(zip(*dert_)), mask__[y], y)  # horizontal clustering - lower row
            derP_ = scan_P_(P_, _P_)  # tests for x overlap between Ps, calls comp_slice

            Pd_ = form_Pd_(P_)  # form Pds within Ps
            derPd_ = scan_Pd_(P_, _P_)  # adds upconnect_ in Pds and calls derPd_2_PP_derPd_, same as derP_2_PP_

            derP__ += derP_; derPd__ += derPd_  # frame of derPs
            P__ += P_; Pd__ += Pd_
            _P_ = P_  # set current lower row P_ as next upper row _P_

        form_PP_shell(blob, derP__, P__, derPd__, Pd__, fPPd)  # form PPs in blob or in FPP

    # draw PPs and FPPs
    if not isinstance(blob, CPP):
        draw_PP_(blob)


def form_P_(idert_, mask_, y):  # segment dert__ into P__, in horizontal ) vertical order
    '''
    sums dert params within Ps and increments L: horizontal length.
    '''
    P_ = []  # rows of derPs
    dert_ = [list(idert_[0])]  # get first dert from idert_ (generator/iterator)
    _mask = mask_[0]  # mask bit per dert
    if ~_mask:
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert_[0]; L = 1; x0 = 0  # initialize P params with first dert

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # pixel mask

        if mask:  # masks: if 1,_0: P termination, if 0,_1: P initialization, if 0,_0: P accumulation:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma), L=L, x0=x0, dert_=dert_, y=y)
                P_.append(P)
        else:  # dert is not masked
            if _mask:  # _dert is masked, initialize P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; L = 1; x0 = x; dert_ = [dert]
            else:
                I += dert[0]  # _dert is not masked, accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                Dy += dert[1]
                Dx += dert[2]
                G += dert[3]
                M += dert[4]
                Dyy += dert[5]
                Dyx += dert[6]
                Dxy += dert[7]
                Dxx += dert[8]
                Ga += dert[9]
                Ma += dert[10]
                L += 1
                dert_.append(dert)
        _mask = mask

    if ~_mask:  # terminate last P in a row
        P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma), L=L, x0=x0, dert_=dert_, y=y)
        P_.append(P)

    return P_

def form_Pd_(P_):
    '''
    form Pd s across P's derts using Dx sign
    '''
    Pd__ = []
    for iP in P_:
        if (iP.downconnect_cnt>0) or (iP.upconnect_):  # form Pd s if at least one connect in P, else they won't be compared
            P_Ddx = 0  # sum of Ddx across Pd s
            P_Mdx = 0  # sum of Mdx across Pd s
            Pd_ = []   # Pds in P
            _dert = iP.dert_[0]  # 1st dert
            dert_ = [_dert]
            I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = _dert; L = 1; x0 = iP.x0  # initialize P params with first dert
            _sign = _dert[2] > 0
            x = 1  # relative x within P

            for dert in iP.dert_[1:]:
                sign = dert[2] > 0
                if sign == _sign: # same Dx sign
                    I += dert[0]  # accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                    Dy += dert[1]
                    Dx += dert[2]
                    G += dert[3]
                    M += dert[4]
                    Dyy += dert[5]
                    Dyx += dert[6]
                    Dxy += dert[7]
                    Dxx += dert[8]
                    Ga += dert[9]
                    Ma += dert[10]
                    L += 1
                    dert_.append(dert)

                else:  # sign change, terminate P
                    P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma),
                           L=L, x0=x0, dert_=dert_, y=iP.y, sign=_sign, Pm=iP)
                    if Dx > ave_Dx:
                        # cross-comp of dx in P.dert_
                        comp_dx(P); P_Ddx += P.Dert.Ddx; P_Mdx += P.Dert.Mdx
                    Pd_.append(P)
                    # reinitialize params
                    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; x0 = iP.x0+x; L = 1; dert_ = [dert]

                _sign = sign
                x += 1
            # terminate last P
            P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma),
                   L=L, x0=x0, dert_=dert_, y=iP.y, sign=_sign, Pm=iP)
            if Dx > ave_Dx:
                comp_dx(P); P_Ddx += P.Dert.Ddx; P_Mdx += P.Dert.Mdx
            Pd_.append(P)
            # update Pd params in P
            iP.Pd_ = Pd_; iP.Dert.Ddx = P_Ddx; iP.Dert.Mdx = P_Mdx
            Pd__ += Pd_

    return Pd__


def scan_P_(P_, _P_):  # test for x overlap between Ps, call comp_slice

    derP_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            # test for x overlap between P and _P in 8 directions
            if (P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0):  # all Ps here are positive

                fcomp = [1 for derP in P.upconnect_ if P is derP.P]  # upconnect could be derP or dirP
                if not fcomp:
                    derP = comp_slice_full(_P, P)  # form vertical and directional derivatives
                    derP_.append(derP)
                    P.upconnect_.append(derP)
                    _P.downconnect_cnt += 1

            elif (P.x0 + P.L) < _P.x0:  # stop scanning the rest of lower P_ if there is no overlap
                break
    return derP_


def scan_Pd_(P_, _P_):  # test for x overlap between Pds

    derPd_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            for Pd in P.Pd_: # lower row Pds
                for _Pd in _P.Pd_: # upper row Pds
                    # test for same sign & x overlap between Pd and _Pd in 8 directions
                    if (Pd.x0 - 1 < (_Pd.x0 + _Pd.L) and (Pd.x0 + Pd.L) + 1 > _Pd.x0) and (Pd.sign == _Pd.sign):

                        fcomp = [1 for derPd in Pd.upconnect_ if Pd is derPd.P]  # upconnect could be derP or dirP
                        if not fcomp:
                            derPd = comp_slice_full(_Pd, Pd)
                            derPd_.append(derPd)
                            Pd.upconnect_.append(derPd)
                            _Pd.downconnect_cnt += 1

                    elif (Pd.x0 + Pd.L) < _Pd.x0:  # stop scanning the rest of lower P_ if there is no overlap
                        break
    return derPd_


def form_PP_shell(blob, derP__, P__, derPd__, Pd__, fPPd):
    '''
    form vertically contiguous patterns of patterns by the sign of derP, in blob or in FPP
    '''
    if not isinstance(blob, CPP):  # input is blob

        blob.derP__ = derP__; blob.P__ = P__
        blob.derPd__ = derPd__; blob.Pd__ = Pd__
        if fPPd:
            derP_2_PP_(blob.derP__, blob.PPdm_, 1, 1)   # cluster by derPm dP sign
            derP_2_PP_(blob.derPd__, blob.PPdd_, 1, 1)  # cluster by derPd dP sign
        else:
            derP_2_PP_(blob.derP__, blob.PPmm_, 1, 0)   # cluster by derPm mP sign
            derP_2_PP_(blob.derPd__, blob.PPmd_, 1, 0)  # cluster by derPd mP sign

        # assign spliced_PP after forming all PPs and FPPs
        PPs_ = [blob.PPdm_,blob.PPdd_,blob.PPmm_,blob.PPmd_]
        for PP_ in PPs_:
            for PP in PP_:
                # check 'if FPP' to save on below?
                # splice FPP with connected PPs:
                for derP in PP.xflip_derP_:  # check derPs where flip_val changed sign
                    _P = derP._P
                    if _P.derP.PP not in PP.xflip_derP_PP_:  # add _PP to P's PP
                        PP.xflip_derP_PP_.append(_P.derP.PP)
#                    if FPP not in _P.derP.PP.xflip_derP_PP_:  # add PP to _P's PP
#                        _P.derP.PP.xflip_derP_PP_.append(PP)
    else:
        FPP = blob  # reassign for clarity
        FPP.derPf__ = derP__; FPP.Pf__ = P__
        FPP.derPdf__ = derPd__; FPP.Pdf__ = Pd__
        if fPPd:
            derP_2_PP_(FPP.derPf__, FPP.PPdmf_, 0, 1)   # cluster by derPmf dP sign
            derP_2_PP_(FPP.derPdf__, FPP.PPddf_, 0, 1)  # cluster by derPdf dP sign
        else:
            derP_2_PP_(FPP.derPf__, FPP.PPmmf_, 0, 0)   # cluster by derPmf mP sign
            derP_2_PP_(FPP.derPdf__, FPP.PPmdf_, 0, 0)  # cluster by derPdf mP sign


def derP_2_PP_(derP_, PP_, fflip, fPPd):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in reversed(derP_):  # bottom-up to follow upconnects, derP is stored top-down
        if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call
            PP = CPP(Dert=CDert(), derPP=CderP())  # init
            accum_PP(PP,derP)

            if derP._P.upconnect_:  # derP has upconnects
                upconnect_2_PP_(derP, PP_, fflip, fPPd)  # form PPs across _P upconnects
            else:
                if (derP.PP.Dert.flip_val > flip_ave_FPP) and fflip:
                    flip_FPP(derP.PP)
                PP_.append(derP.PP)


def upconnect_2_PP_(iderP, PP_, fflip, fPPd):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP__:  # derP should not in current iPP derP_ list, but this may occur after the PP merging

            if (derP.P.Dert.flip_val>0 and iderP.P.Dert.flip_val>0 and iderP.PP.Dert.flip_val>0):
                # upconnect derP has different FPP, merge them
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else: # accumulate derP to current FPP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)
            # not FPP
            else:
                if fPPd: same_sign = (iderP.dP > 0) == (derP.dP > 0)  # comp dP sign
                else: same_sign = (iderP.mP > 0) == (derP.mP > 0)  # comp mP sign

                if same_sign and not (iderP.P.Dert.flip_val>0) and not (derP.P.Dert.flip_val>0):  # upconnect derP has different PP, merge them
                        if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                            merge_PP(iderP.PP, derP.PP, PP_)
                        else:  # accumulate derP in current PP
                            accum_PP(iderP.PP, derP)
                            confirmed_upconnect_.append(derP)
                elif not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has FPP/PP
                    PP = CPP(Dert=CDert(), derPP=CderP())
                    accum_PP(PP,derP)
                    derP.P.downconnect_cnt = 0  # reset downconnect count for root derP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fflip, fPPd)  # recursive compare sign of next-layer upconnects

            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                if (derP.PP.Dert.flip_val > flip_ave_FPP) and fflip:
                    flip_FPP(derP.PP)
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if not iderP.P.downconnect_cnt:
        if (iderP.PP.Dert.flip_val > flip_ave_FPP) and fflip:
            flip_FPP(iderP.PP)
        PP_.append(iderP.PP)  # iPP is terminated after all upconnects are checked


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP__:
        if derP not in _PP.derP__:
            _PP.derP__.append(derP)
            derP.PP = _PP  # update reference

            Dert = derP.P.Dert
            # accumulate Dert param of derP
            _PP.Dert.accumulate(I=Dert.I, Dy=Dert.Dy, Dx=Dert.Dx, G=Dert.G, M=Dert.M, Dyy=Dert.Dyy, Dyx=Dert.Dyx, Dxy=Dert.Dxy, Dxx=Dert.Dxx,
                               Ga=Dert.Ga, Ma=Dert.Ma, Mdx=Dert.Mdx, Ddx=Dert.Ddx, flip_val=Dert.flip_val)

            # accumulate if PP' derP not in _PP
            _PP.derPP.accumulate(mP=derP.mP, dP=derP.dP, mx=derP.mx, dx=derP.dx,
                                 mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                                 mDy=derP.mDy, dDy=derP.dDy)

    for splice_derP in PP.xflip_derP_:
        if splice_derP not in _PP.xflip_derP_:
            _PP.xflip_derP_.append(splice_derP)

    if PP in PP_:
        PP_.remove(PP)  # remove merged PP


def flip_FPP(FPP):
    '''
    flip derts of FPP and call again slice_blob to get PPs of FPP
    '''
    # get box from P and P
    x0 = min(min([derP.P.x0 for derP in FPP.derP__]), min([derP._P.x0 for derP in FPP.derP__]))
    xn = max(max([derP.P.x0+derP.P.L for derP in FPP.derP__]), max([derP._P.x0+derP._P.L for derP in FPP.derP__]))
    y0 = min(min([derP.P.y for derP in FPP.derP__]), min([derP._P.y for derP in FPP.derP__]))
    yn = max(max([derP.P.y for derP in FPP.derP__]), max([derP._P.y for derP in FPP.derP__])) +1  # +1 because yn is not inclusive
    FPP.box = [y0,yn,x0,xn]
    # init empty derts, 11 params each: p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma
    dert__ = [np.zeros((yn-y0, xn-x0)) for _ in range(11)]
    mask__ = np.ones((yn-y0, xn-x0)).astype('bool')

    # fill empty dert with current FPP derts
    for derP in FPP.derP__:
        # _P
        for _x, _dert in enumerate(derP._P.dert_):
            for i, _param in enumerate(_dert):
                dert__[i][derP._P.y-y0, derP._P.x0-x0+_x] = _param
                mask__[derP._P.y-y0, derP._P.x0-x0+_x] = False
        # P
        for x, dert in enumerate(derP.P.dert_):
            for j, param in enumerate(dert):
                dert__[j][derP.P.y-y0, derP.P.x0-x0+x] = param
                mask__[derP.P.y-y0, derP.P.x0-x0+x] = False
    # flip dert__
    flipped_dert__ = [np.rot90(dert) for dert in dert__]
    flipped_mask__ = np.rot90(mask__)
    flipped_dert__[1],flipped_dert__[2] = \
    flipped_dert__[2],flipped_dert__[1]  # swap dy and dx in derts, always flipped in FPP
    FPP.dert__ = flipped_dert__
    FPP.mask__ = flipped_mask__
    # form PP_ in flipped FPP
    slice_blob(FPP, verbose=True)


def flip_eval_blob(blob):

    # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: preferential comp over low G
    horizontal_bias = (blob.box[3] - blob.box[2]) / (blob.box[1] - blob.box[0])  \
                    * (abs(blob.Dy) / abs(blob.Dx))

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):
        blob.fflip = 1  # rotate 90 degrees for scanning in vertical direction
        # swap blob Dy and Dx:
        Dy=blob.Dy; blob.Dy = blob.Dx; blob.Dx = Dy
        # rotate dert__:
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)
        # swap dert dys and dxs:
        blob.dert__ = list(blob.dert__)  # convert to list since param in tuple is immutable
        blob.dert__[1], blob.dert__[2] = \
        blob.dert__[2], blob.dert__[1]


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})

def accum_PP(PP, derP):  # accumulate derP params in PP

    Dert = derP.P.Dert
    # accumulate Dert params
    ''' use:
    for param, PP_param in zip(Dert, PP.Dert):
        PP_param+=param
    ?
    '''
    PP.Dert.accumulate(I=Dert.I, Dy=Dert.Dy, Dx=Dert.Dx, G=Dert.G, M=Dert.M, Dyy=Dert.Dyy, Dyx=Dert.Dyx, Dxy=Dert.Dxy, Dxx=Dert.Dxx,
                     Ga=Dert.Ga, Ma=Dert.Ma, Mdx=Dert.Mdx, Ddx=Dert.Ddx, flip_val=Dert.flip_val)
    # accumulate derP params
    PP.derPP.accumulate(mP=derP.mP, dP=derP.dP, mx=derP.mx, dx=derP.dx, mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                        mDy=derP.mDy, dDy=derP.dDy)
    PP.derP__.append(derP)

    derP.PP = PP  # update reference

    if derP.fxflip: # add splice point
        PP.xflip_derP_.append(derP)

def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Dert.Ddx = Ddx
    P.Dert.Mdx = Mdx


def comp_slice(_P, P, _derP_):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    s, x0, Dx, Dy, G, M, L, Ddx, Mdx = P.sign, P.x0, P.Dert.Dx, P.Dert.Dy, P.Dert.G, P.Dert.M, P.L, P.Dert.Ddx, P.Dert.Mdx  # params per comp branch
    _s, _x0, _Dx, _Dy, _G, _M, _dX, _L, _Ddx, _Mdx = _P.sign, _P.x0, _P.Dert.Dx, _P.Dert.Dy, _P.Dert.G, _P.Dert.M, _P.dX, _P.L, _P.Dert.Ddx, _P.Dert.Mdx

    dX = (x0 + (L-1) / 2) - (_x0 + (_L-1) / 2)  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?
    hyp = np.hypot(dX, 1)  # ratio of local segment of long (vertical) axis to dY = 1

    L /= hyp  # orthogonal L is reduced by hyp
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    M /= hyp  # orthogonal M is reduced by hyp
    dM = M - _M; mM = min(M, _M)  # use abs M?  no Mx, My: non-core, lesser and redundant bias?

    dP = ddX + dL + dM  # -> directional PPd, equal-weight params, no rdn?
    mP = mdX + mL + mM  # -> complementary PPm, rdn *= Pd | Pm rolp?
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    P.Dert.flip_val = (dX * (P.Dert.Dy / (P.Dert.Dx+.001)) - flip_ave)  # avoid division by zero

    derP = CderP(P=P, _P=_P, mP=mP, dP=dP, dX=dX, mL=mL, dL=dL)
    P.derP = derP

    # if flip value>0 AND positive mP (predictive value) AND flip_val sign changed AND _P.derP is derP: exclude 1st row Ps
    if (P.Dert.flip_val>0) and (derP.mP >0) and ((P.Dert.flip_val>0) != (_P.Dert.flip_val>0)) and (isinstance(_P.derP, CderP)):
        derP.fxflip = 1

    return derP


def comp_slice_full(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    s, x0, Dx, Dy, G, M, L, Ddx, Mdx = P.sign, P.x0, P.Dert.Dx, P.Dert.Dy, P.Dert.G, P.Dert.M, P.L, P.Dert.Ddx, P.Dert.Mdx
    # params per comp branch, add angle params
    _s, _x0, _Dx, _Dy, _G, _M, _dX, _L, _Ddx, _Mdx = _P.sign, _P.x0, _P.Dert.Dx, _P.Dert.Dy, _P.Dert.G, _P.Dert.M, _P.dX, _P.L, _P.Dert.Ddx, _P.Dert.Mdx

    dX = (x0 + (L-1) / 2) - (_x0 + (_L-1) / 2)  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        xn = x0 + L - 1
        _xn = _x0 + _L - 1
        mX = min(xn, _xn) - max(x0, _x0)  # overlap = abs proximity: summed binary x match
        rX = dX / mX if mX else dX*2  # average dist / prox, | prox / dist, | mX / max_L?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?

    if dX * P.Dert.G > ave_ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
        # diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/orthogonalization.png
        # Long axis is a curve of connections between ave_xs: mid-points of consecutive Ps.

        # Ortho virtually rotates P to connection-orthogonal direction:
        hyp = np.hypot(dX, 1)  # ratio of local segment of long (vertical) axis to dY = 1
        L = L / hyp  # orthogonal L
        # combine derivatives in proportion to the contribution of their axes to orthogonal axes:
        # contribution of Dx should increase with hyp(dX,dY=1), this is original direction of Dx:
        Dy = (Dy / hyp + Dx * hyp) / 2  # estimated along-axis D
        Dx = (Dy * hyp + Dx / hyp) / 2  # estimated cross-axis D
        '''
        alternatives:
        oDy = (Dy * hyp - Dx / hyp) / 2;  oDx = (Dx / hyp + Dy * hyp) / 2;  or:
        oDy = hypot( Dy / hyp, Dx * hyp);  oDx = hypot( Dy * hyp, Dx / hyp)
        '''
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # use abs M?  no Mx, My: non-core, lesser and redundant bias?
    # no comp G: Dy, Dx are more specific:
    dDx = Dx - _Dx  # same-sign Dx if Pd
    mDx = min(abs(Dx), abs(_Dx))
    if Dx > 0 != _Dx > 0: mDx = -mDx
    # min is value distance for opposite-sign comparands, vs. value overlap for same-sign comparands
    dDy = Dy - _Dy  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI
    mDy = min(abs(Dy), abs(_Dy))
    if (Dy > 0) != (_Dy > 0): mDy = -mDy

    dDdx, dMdx, mDdx, mMdx = 0, 0, 0, 0
    if P.dxdert_ and _P.dxdert_:  # from comp_dx
        fdx = 1
        dDdx = Ddx - _Ddx
        mDdx = min( abs(Ddx), abs(_Ddx))
        if (Ddx > 0) != (_Ddx > 0): mDdx = -mDdx
        # Mdx is signed:
        dMdx = min( Mdx, _Mdx)
        mMdx = -min( abs(Mdx), abs(_Mdx))
        if (Mdx > 0) != (_Mdx > 0): mMdx = -mMdx
    else:
        fdx = 0
    # coeff = 0.7 for semi redundant parameters, 0.5 for fully redundant parameters:
    dP = ddX + dL + 0.7*(dM + dDx + dDy)  # -> directional PPd, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?
    if fdx: dP += 0.7*(dDdx + dMdx)

    mP = mdX + mL + 0.7*(mM + mDx + mDy)  # -> complementary PPm, rdn *= Pd | Pm rolp?
    if fdx: mP += 0.7*(mDdx + mMdx)
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    P.Dert.flip_val = (dX * (P.Dert.Dy / (P.Dert.Dx+.001)) - flip_ave)  # avoid division by zero

    derP = CderP(P=P, _P=_P, mP=mP, dP=dP, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy)
    P.derP = derP

    # if flip value>0 AND positive mP (predictive value) AND flip_val sign changed AND _P.derP is derP: exclude 1st row Ps
    if (P.Dert.flip_val>0) and (derP.mP >0) and ((P.Dert.flip_val>0) != (_P.Dert.flip_val>0)) and (isinstance(_P.derP, CderP)):
        derP.fxflip = 1

    if fdx:
        derP.fdx=1; derP.dDdx=dDdx; derP.mDdx=mDdx; derP.dMdx=dMdx; derP.mMdx=mMdx

    '''
    min comp for rotation: L, Dy, Dx, no redundancy?
    mParam weighting by relative contribution to mP, /= redundancy?
    div_f, nvars: if abs dP per PPd, primary comp L, the rest is normalized?
    '''
    return derP

''' radial comp extension for co-internal blobs:
    != sign comp x sum( adj_blob_) -> intra_comp value, isolation value, cross-sign merge if weak, else:
    == sign comp x ind( adj_adj_blob_) -> same-sign merge | composition:
    borrow = adj_G * rA: default sum div_comp S -> relative area and distance to adjj_blob_
    internal sum comp if mA: in thin lines only? comp_norm_G or div_comp_G -> rG?
    isolation = decay + contrast:
    G - G * (rA * ave_rG: decay) - (rA * adj_G: contrast, = lend | borrow, no need to compare vG?)
    if isolation: cross adjj_blob composition eval,
    else:         cross adjj_blob merge eval:
    blob merger if internal match (~raG) - isolation, rdn external match:
    blob compos if external match (~rA?) + isolation,
    Also eval comp_slice over fork_?
    rng+ should preserve resolution: rng+_dert_ is dert layers,
    rng_sum-> rng+, der+: whole rng, rng_incr-> angle / past vs next g,
    rdn Rng | rng_ eval at rng term, Rng -= lost coord bits mag, always > discr?
'''