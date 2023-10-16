# Cycahvore
 Cython wrapper for CAHVORE camera modules from VICAR.
 


### Introduction

Cycahvore (*sigh*-*cah*-*vor*) is a cython wrapper for the C 'cahvor' camera models from JPL's VICAR software.

It currently wraps a subset of the functions available in the C library for CAHV, CAHVOR, CAHVORE camera models,
assuming contiguous numpy arrays as inputs. Some functions are also provided with vectorized wrappers to efficiently loop over many coordinates in N-D numpy arrays.
