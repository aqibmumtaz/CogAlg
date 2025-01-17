'''
Comp_slice is a terminal fork of intra_blob.
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat or low-G blobs.
Vectorization is clustering of Ps + their derivatives into PPs: patterns of Ps that describe an edge.
'''

from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = 2000
div_ave = 200
ave_dX = 10  # difference between median x coords of consecutive Ps

class CP(ClusterStructure):
    # Dert: summed pixel values and pixel-level derivatives:
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
    L = int
    x0 = int
    y = int  # for visualization only
    sign = NoneType  # sign of gradient deviation
    dert_ = list   # array of pixel-level derts: (p, dy, dx, g, m), extended in intra_blob
    gdert_ = list
    Dg = int
    Mg = int
    derP_ = list
    upconnect_ = list
    downconnect_cnt = int


class CderP(ClusterStructure):

    Pi = object  # P instance, for accumulation: CderP.Pi.I += 1, etc.
    # this would be derPP in accum_PP, no need here?
    Pm = int
    Pd = int
    mx = int
    dx = int
    mL = int
    dL = int
    mDx = int
    dDx = int
    mDy = int
    dDy = int
    mDg = int
    dDg = int
    mMg = int
    dMg = int
    P = object   # lower comparand
    _P = object  # higher comparand
    PP = object  # contains this derP

class CPP(ClusterStructure):

    derPP = object  # set of derP params accumulated in PP
    derP_ = list
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType

# Functions:

# leading '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
# trailing '_' denotes array name, vs. same-name elements of that array. '__' is a 2D array
# leading 'f' denotes flag

def slice_blob(blob, verbose=False):
    '''
    Slice_blob converts selected smooth-edge blobs (high G, low Ga) into sliced blobs,
    adding horizontal blob slices: Ps or 1D patterns
    '''

    flip_eval(blob)
    dert__ = blob.dert__
    mask__ = blob.mask__
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")
    P__ = []
    derP__ = []

    zip_dert__ = zip(*dert__)
    _P_ = form_P_(list(zip(*next(zip_dert__))), mask__[0], 0)  # 1st row
    P__ += _P_  # frame of Ps

    for y, dert_ in enumerate(zip_dert__, start=1):  # scan top down
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y], y)  # horizontal clustering - lower row
        derP_ = scan_P_(P_, _P_)  # test x overlap between Ps, call comp_slice
        derP__ += derP_  # frame of derPs
        _P_ = P_  # set current lower row P_ as next upper row _P_

    blob.P__ = P__
    blob.derP__ = derP__
    derP_2_PP_(blob.derP__, blob.PP_)


def form_P_(idert_, mask_, y):  # segment dert__ into P__, in horizontal ) vertical order
    '''
    sums dert params within Ps and increments L: horizontal length.
    '''
    P_ = [] # rows of derPs
    dert_ = [list(idert_[0])]  # get first dert from idert_ (generator/iterator)
    _mask = mask_[0]  # mask bit per dert
    if ~_mask:
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert_[0]; L = 1; x0 = 0  # initialize P params with first dert

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # pixel mask

        if mask:  # masks = 1,_0: P termination, 0,_1: P initialization, 0,_0: P accumulation:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_, y=y)
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
        P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_, y=y)
        P_.append(P)

    return P_


def scan_P_(P_, _P_):  # test for x overlap between Ps, call comp_slice

    derP_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            # test for x overlap between P and _P in 8 directions:
            if P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0:  # all Ps here are positive

                fcomp = [1 for derP in P.upconnect_ if P is derP.P]
                if not fcomp:
                    derP = comp_slice(P, _P)  # form vertical derivatives
                    derP_.append(derP)
                    _P.downconnect_cnt += 1
                    P.upconnect_.append(derP)

            elif (P.x0 + P.L) < _P.x0:  # stop scanning the rest of lower P_ if there is no overlap
                break

    return derP_


def flip_eval(blob):

    # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: , preferential comp over low G
    horizontal_bias = (blob.box[3] - blob.box[2]) / (blob.box[1] - blob.box[0]) \
                      * (abs(blob.Dy) / abs(blob.Dx))

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):
        blob.fflip = 1  # rotate 90 degrees for scanning in vertical direction
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


def comp_slice(P, _P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    s, x0, G, M, Dx, Dy, L, Dg, Mg = P.sign, P.x0, P.G, P.M, P.Dx, P.Dy, P.L, P.Dg, P.Mg
    # params per comp branch, add angle params, ext: X, new: L,
    # no input I comp in top dert?
    _s, _x0, _G, _M, _Dx, _Dy, _L, _Dg, _Mg = _P.sign, _P.x0, _P.G, _P.M, _P.Dx, _P.Dy, _P.L, _P.Dg, _P.Mg
    '''
    redefine Ps by dx in dert_, rescan dert by input P d_ave_x: skip if not in blob?
    '''
    xn = x0 + L-1;  _xn = _x0 + _L-1
    # to be revised:
    mX = min(xn, _xn) - max(x0, _x0)  # overlap: abs proximity, cumulative binary positional match | miss:
    _dX = (xn - L/2) - (_xn - _L/2)
    dX = abs(x0 - _x0) + abs(xn - _xn)  # offset, or max_L - overlap: abs distance?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        rX = dX / mX if mX else dX*2  # average dist/prox, | prox/dist, | mX / max_L?

    ave_dx = (x0 + (L-1)/2) - (_x0 + (_L-1)/2)  # d_ave_x, median vs. summed, or for distant-P comp only?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    # param correlations: dX-> L, ddX-> dL, neutral to Dx: mixed with anti-correlated oDy?
    '''
    if ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
        Long axis is a curve, consisting of connections between mid-points of consecutive Ps.
        Ortho virtually rotates each P to make it orthogonal to its connection:
        hyp = hypot(dX, 1)  # long axis increment (vertical distance), to adjust params of orthogonal slice:
        L /= hyp
        # re-orient derivatives by combining them in proportion to their decomposition on new axes:
        Dx = (Dx * hyp + Dy / hyp) / 2  # no / hyp: kernel doesn't matter on P level?
        Dy = (Dy / hyp - Dx * hyp) / 2  # estimated D over vert_L
    '''
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # no Mx, My: non-core, lesser and redundant bias?

    dDx = abs(Dx) - abs(_Dx); mDx = min(abs(Dx), abs(_Dx))  # same-sign Dx in vxP
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI

    # gdert param comparison, if not fPP, values would be 0
    dMg = Mg - _Mg; mMg = min(Mg, _Mg)
    dDg = Dg - _Dg; mDg = min(Dg, _Dg)

    Pd = ddX + dL + dM + dDx + dDy + dMg + dDg # -> directional dPP, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?  G = hypot(Dy, Dx) for 2D structures comp?
    Pm = mX + mL + mM + mDx + mDy + mMg + mDg # -> complementary vPP, rdn *= Pd | Pm rolp?

    derP = CderP(P=P, _P=_P, Pm=Pm, Pd=Pd, mX=mX, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy, mDg=mDg, dDg=dDg, mMg=mMg, dMg=dMg)
    # div_f, nvars

    return derP

''' Positional miss is positive: lower filters, no match: always inverse miss?

    skip to prediction limits: search for termination that defines and borrows from P,
    form complemented ) composite Ps: ave proximate projected match cross sign ) composition level:
     
    comp @ ave m: balance between current m and higher-composition m, same for projected m?
    for hierarchical vs. lateral novelty expansion: higher-order projection?
          
    1Le: fixed binary Cf, 2Le: skip to individual integer Cf, variable pattern L vs. pixel res?
    edge is more concentrated than flat for stable shape -> stable contents patterns?
    differential feedback per target level: @filters, but not pattern
    
    radial comp extension for co-internal blobs:
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

def derP_2_PP_(derP_, PP_):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in reversed(derP_):  # bottom-up to follow upconnects, derP is stored top-down
        if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call

            PP = CPP(derPP=derP, derP_= [derP])  # init
            derP.PP = PP

            if derP._P.upconnect_:  # derP has upconnects
                upconnect_2_PP_(derP, PP_)  # form PPs across _P upconnects
            else:
                PP_.append(derP.PP)
    return PP_


def upconnect_2_PP_(iderP, PP_):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []
    for derP in iderP._P.upconnect_:  # potential upconnects from previous call

        if derP not in iderP.PP.derP_:  # derP should not in current iPP derP_ list, but this may occur after the PP merging
            if (iderP.Pm > 0) == (derP.Pm > 0):  # no sign change, accumulate PP

                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):  # upconnect derP has different PP, merge them
                    merge_PP(iderP.PP, derP.PP, PP_)
                else: # accumulate derP to current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)

            elif not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has PP

                derP.PP = CPP(derPP=derP, derP_=[derP])  # init
                derP.P.downconnect_cnt = 0  # reset downconnect count for root derP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_)  # recursive compare sign of next-layer upconnects

            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if not iderP.P.downconnect_cnt:
        PP_.append(iderP.PP)  # iPP termination after all upconnects are checked


def accum_PP(PP, derP):

    PP.derPP.accumulate(Pm=derP.Pm, Pd=derP.Pd, mx=derP.mx, dx=derP.dx, mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                          mDy=derP.mDy, dDy=derP.dDy, mDg=derP.mDg, dDg=derP.dDg, mMg=derP.mMg, dMg=derP.dMg)
    PP.derP_.append(derP)
    derP.PP = PP


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP_:
        if derP not in _PP.derP_:  # there may be overlapping derPs between PPs
            _PP.derP_.append(derP)
            derP.PP = _PP  # update reference

            # accumulate if PP' derP not in _PP
            _PP.derPP.accumulate(Pm=derP.Pm, Pd=derP.Pd, mx=derP.mx, dx=derP.dx,
                                 mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                                 mDy=derP.mDy, dDy=derP.dDy, mDg=derP.mDg,
                                 dDg=derP.dDg, mMg=derP.mMg, dMg=derP.dMg)
    if PP in PP_:
        PP_.remove(PP)  # remove the merged PP
