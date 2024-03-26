# cython: language_level=3, boundscheck=False, emit_code_comments=True, embedsignature=True, initializedcheck=False

from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
np.import_array()
from .cimport cmod