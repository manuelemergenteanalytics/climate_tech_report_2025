import numpy as np


def winsorize(x, p=0.95):
hi = np.quantile(x, p)
return np.clip(x, None, hi)