import cv2
import numpy as np
from frame_2D_alg_classes import filters
import matplotlib.pyplot as plt
# ***************************************************** MISCELLANEOUS FUNCTIONS *****************************************
# Functions:
# -get_filters()
# ***********************************************************************************************************************
def get_filters(obj):
    " imports all variables in filters.py "
    str_ = [item for item in dir(filters) if not item.startswith("__")]
    for str in str_:
        var = getattr(filters, str)
        obj[str] = var
    # ---------- get_filters() end --------------------------------------------------------------------------------------
# ***************************************************** MISCELLANEOUS FUNCTIONS END *************************************