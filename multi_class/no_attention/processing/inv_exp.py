# -*- coding: utf-8 -*-

import math
import numpy as np

def inv_exp(X, Y):    

    skeleton=X.dot(Y.T)
    tr=abs(skeleton.trace())
    if tr>1:
        tr=1
    teta_invexp=math.acos(tr)
    if (math.sin(teta_invexp)<0.0001):
        teta_invexp=0.1

    invExp=(teta_invexp/math.sin(teta_invexp)) * (Y - (math.cos(teta_invexp))*X)     
    np_inv = np.array(invExp)

    return invExp