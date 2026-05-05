import numpy as np

def quantile_bins(x, n):
    """
    Adaptive binning with at least n samples per bin.

    Parameters
    ----------
    x : array-like
        1D data
    n : int
        Minimum number of samples per bin

    Returns
    -------
    bins : ndarray
        Bin edges
    """
    x = np.sort(np.asarray(x))
    if n <= 0:
        raise ValueError("n must be positive")

    bins = x[::n]
    if bins[-1] != x[-1]:
        bins = np.append(bins, x[-1])

    return bins

def fixed_width_bins_min_count(x, n):
    """
    Smallest uniform bin width such that every bin has at least n samples.

    Parameters
    ----------
    x : array-like
        1D data
    n : int
        Minimum samples per bin

    Returns
    -------
    bins : ndarray
        Uniform bin edges
    width : float
        Bin width
    """
    x = np.sort(np.asarray(x))
    if n <= 0 or n >= len(x):
        raise ValueError("n must be between 1 and len(x)-1")

    width = np.max(x[n:] - x[:-n])
    bins = np.arange(x.min(), x.max() + width, width)

    return bins, width


def adaptive_bins_max_width(x, n, w_max):
    """
    Adaptive binning with minimum count and maximum width constraint.

    Parameters
    ----------
    x : array-like
        1D data
    n : int
        Minimum samples per bin
    w_max : float
        Maximum allowed bin width

    Returns
    -------
    bins : ndarray
        Bin edges
    """
    x = np.sort(np.asarray(x))
    if n <= 0:
        raise ValueError("n must be positive")
    if w_max <= 0:
        raise ValueError("w_max must be positive")

    bins = [x[0]]
    i = 0
    N = len(x)

    while i < N:
        j = min(i + n, N - 1)

        while j < N - 1 and x[j] - x[i] < w_max:
            j += 1

        bins.append(x[j])
        i = j

    return np.array(bins)

def bin_counts(x, bins):
    """
    Return counts per bin.
    """
    counts, _ = np.histogram(x, bins=bins)
    return counts

# quantile_bins	Adaptive	≥ n samples
# fixed_width_bins_min_count	Uniform	≥ n samples
# adaptive_bins_max_width	Adaptive	≥ n + width limit


def find_bin_index(x0, bins):
    """
    Return index i such that x0 is in [bins[i], bins[i+1]).
    If x0 is outside range, return closest bin.
    """
    bins = np.asarray(bins)

    if x0 <= bins[0]:
        return 0
    if x0 >= bins[-1]:
        return len(bins) - 2

    return np.searchsorted(bins, x0) - 1

def sample_Y_from_bin(x, Y, bins, bin_index, rng=None):
    """
    Randomly select one row from Y whose x falls in the given bin.

    Returns
    -------
    y_sample : ndarray, shape (120,)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x)
    Y = np.asarray(Y)

    lo = bins[bin_index]
    hi = bins[bin_index + 1]

    mask = (x >= lo) & (x < hi)

    if not np.any(mask):
        raise ValueError("Selected bin contains no samples")

    indices = np.where(mask)[0]
    idx = rng.choice(indices)

    return Y[idx]

def sample_Y_at_x0(x, Y, bins, x0):
    i = find_bin_index(x0, bins)
    return sample_Y_from_bin(x, Y, bins, i)


def find_nearest_bin_center(x0, bins):
    """
    Return index of bin whose center is closest to x0.
    """
    bins = np.asarray(bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return np.argmin(np.abs(centers - x0))

# def sample_k_Y_from_bin(x, Y, bins, bin_index, k=1, rng=None, replace=False):
#     """
#     Randomly sample k rows from Y whose x fall in the given bin.

#     Parameters
#     ----------
#     k : int
#         Number of samples
#     replace : bool
#         Sample with replacement if True
#     rng : np.random.Generator
#         For reproducibility
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     x = np.asarray(x)
#     Y = np.asarray(Y)

#     lo = bins[bin_index]
#     hi = bins[bin_index + 1]

#     mask = (x >= lo) & (x < hi)
#     indices = np.where(mask)[0]

#     if len(indices) == 0:
#         raise ValueError("Selected bin contains no samples")

#     if not replace and k > len(indices):
#         raise ValueError("k larger than available samples in bin")

#     chosen = rng.choice(indices, size=k, replace=replace)
#     return Y[chosen]
#     # rng = np.random.default_rng(seed=42)
    
    
#     # # build bins (example: quantile bins)
#     # bins = quantile_bins(x, n=30)
    
#     # # reproducible RNG
#     # rng = np.random.default_rng(123)
    
#     # # query value
#     # x0 = 0.75
    
#     # # nearest-bin-center selection
#     # bin_index = find_nearest_bin_center(x0, bins)
    
#     # # sample k arrays
#     # Y_samples = sample_k_Y_from_bin(
#     #     x, Y,
#     #     bins,
#     #     bin_index,
#     #     k=5,
#     #     replace=False,
#     #     rng=rng
#     # )
    
#     # print(Y_samples.shape)  # (5, 120)


def plot_binned_histogram(x, bins, ax=None, **kwargs):
    """
    Plot 1D histogram with predefined bins.

    Parameters
    ----------
    x : array-like
        Data values
    bins : array-like
        Bin edges
    ax : matplotlib axis, optional
    kwargs :
        Passed to ax.hist()
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(x, bins=bins, edgecolor='black', **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("Counts")

    return ax


def highlight_bin(ax, bins, bin_index, x0=None):
    """
    Highlight a selected bin and optional x0.
    """
    lo = bins[bin_index]
    hi = bins[bin_index + 1]

    ax.axvspan(lo, hi, color='orange', alpha=0.3, label='Selected bin')

    if x0 is not None:
        ax.axvline(x0, color='red', linestyle='--', label='x0')

    ax.legend()


def plot_bin_counts(x, bins, ax=None):
    counts, _ = np.histogram(x, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(centers, counts, width=widths, align='center',
           edgecolor='black')

    ax.set_xlabel("x")
    ax.set_ylabel("Counts")

    return ax

def sample_k_Y_from_bin(
    x, Y, bins, bin_index,
    k=1,
    rng=None
):
    """
    Sample k rows from Y whose x fall in the given bin.
    If k > available samples, sampling is done with replacement.

    Returns
    -------
    Y_samples : ndarray, shape (k, 120)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x)
    Y = np.asarray(Y)

    lo = bins[bin_index]
    hi = bins[bin_index + 1]

    mask = (x >= lo) & (x < hi)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError("Selected bin contains no samples")

    replace = k > len(indices)

    chosen = rng.choice(indices, size=k, replace=replace)
    return Y[chosen]


# def generate_rowwise_gaussian_noise(sigmas):
#     """
#     Generate Gaussian noise where each row has its own diagonal covariance.

#     Parameters
#     ----------
#     sigmas : ndarray, shape (N, 120)
#         Row-wise standard deviations.

#     Returns
#     -------
#     noise : ndarray, shape (N, 120)
#         Generated Gaussian noise.
        
#     Each row i is sampled as:
    
#     ϵi∼N(0, diag(A[i,:]^2))

#     """
#     sigmas = np.asarray(sigmas)

#     if sigmas.ndim != 2:
#         raise ValueError("sigmas must have shape (N, 120)")

#     noise = np.random.randn(*sigmas.shape) * sigmas
#     return noise


def generate_rowwise_gaussian_noise(sigmas, seed=40):
    """
    Generate Gaussian noise where each row has its own diagonal covariance,
    with a fixed random seed.

      Parameters
    ----------
    sigmas : ndarray, shape (N, 120)
        Row-wise standard deviations.

    Returns
    -------
    noise : ndarray, shape (N, 120)
        Generated Gaussian noise.
        
    Each row i is sampled as:
    
    ϵi∼N(0, diag(A[i,:]^2))

    """
    sigmas = np.asarray(sigmas)

    if sigmas.ndim != 2:
        raise ValueError("sigmas must have shape (N, 120)")

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(sigmas.shape) * sigmas
    return noise
