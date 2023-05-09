import numpy as np

# NOTE: temporary functions ripped from old code. To be made compatible with Geolib
# functions


def calc_ic(qc, rf) -> np.ndarray:
    """
    Calculate non-normalized IC values (I_SBT in Robertson 2010). The non-normalized
    variant does not require calculations of stresses to normalize with and hence no PWP
    data is required.

    Please note the following when using non-normalized IC values:

    "The non-normalized SBT index (ISBT) is essentially the same as the normalized SBTn
    index (Ic) but only uses the basic CPT measurements. In general, the normalized Ic
    provides more reliable identification of SBT than the non-normalized ISBT, but when
    the insitu vertical effective stress is between 50 kPa to 150 kPa there is often
    little difference between normalized and non-normalized SBT."

    Parameters
    ----------
    qc : np.ndarray
        Cone resistance values
    rf : np.ndarray
        Friction number

    Returns
    -------
    np.ndarray
        Non-normalized IC
    """
    ic = np.sqrt((3.47 - np.log10(qc / 0.1)) ** 2 + (np.log10(rf) + 1.22) ** 2)
    return ic


# TODO numpy searchsort
def calc_lithology(ic, qc, rf) -> np.ndarray:
    boundaries = [1.6, 2.0, 2.2, 2.6, 2.95, 3.6]
    lith = np.full_like(ic, "NBE", dtype="<U3")
    lith[ic < boundaries[0]] = "Z"
    lith[(ic >= boundaries[0]) & (ic < boundaries[1])] = "Z"
    lith[(ic >= boundaries[1]) & (ic < boundaries[2])] = "Z"
    lith[(ic >= boundaries[2]) & (ic < boundaries[3])] = "Z"
    lith[(ic >= boundaries[3]) & (ic < boundaries[4])] = "Kz"
    lith[(ic >= boundaries[4]) & (ic < boundaries[5])] = "K"
    lith[(ic >= boundaries[5]) & (rf > 8)] = "V"
    lith[((rf > 5) & (qc < 1.5)) | (rf > 6)] = "V"
    lith[(ic >= boundaries[5]) & (rf <= 8)] = "Kh"
    return lith
