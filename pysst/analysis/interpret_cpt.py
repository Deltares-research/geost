import numpy as np

# NOTE: temporary functions ripped from old code. To be made compatible with Geolib functions


def calc_ic(qc, rf) -> np.ndarray:
    return np.sqrt((3.47 - np.log10(qc / 0.1)) ** 2 + (np.log10(rf) + 1.22) ** 2)


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
