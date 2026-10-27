"""Standalone functions for the three example notebooks.

Processing and prediction functions are copied from the research project;
all data paths below resolve relative to this file. No original project import
or absolute data path is required. See validation/provenance.json.
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import hilbert, butter, filtfilt, lfilter, correlate, correlation_lags, find_peaks, peak_widths
from scipy.signal.windows import tukey
from scipy.fft import fft, ifft, fftshift
from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PATH_AWS_DATA = DATA / 'aws' / 'AWS_heroy_data_UTC.csv'
PATH_ERA5_GRADIENTS = DATA / 'era5' / 'gradients_0_50m.csv'
_df_aws = None
K_pierce = 0.509 / (8 * np.pi ** 2) ** (1 / 3)
RESP_COMBOS = [('linear', False, 'linear'), ('wpws', True, 'pws')]
GROUPS = {'WNW': ['S084a','S074a','S096a','S092a'],
          'South': ['S068a','S094a','S020a','S019a','S076a']}
SOURCE, REFERENCE = 'S056a', 'S007a'
TZ = 'America/Chicago'
START = pd.Timestamp('2024-04-08 11:00').tz_localize(TZ).tz_convert('UTC').tz_localize(None)
END = pd.Timestamp('2024-04-09 05:00').tz_localize(TZ).tz_convert('UTC').tz_localize(None)


# Functions from src/utils_time.py

def to_utc_naive(t, tz_str=None):
    """Convert any time-like object to a UTC timezone-naive pd.Timestamp.

    Parameters
    ----------
    t : str | datetime.datetime | pd.Timestamp | obspy.UTCDateTime
        Input time in any supported format.
    tz_str : str or None
        IANA timezone name for interpreting naive inputs.
        None → naive input is already UTC.

    Returns
    -------
    pd.Timestamp
        UTC, timezone-naive.

    Examples
    --------
    >>> to_utc_naive('2024-04-08 18:00:00')
    Timestamp('2024-04-08 18:00:00')

    >>> to_utc_naive('2024-04-08 13:00:00', tz_str='America/Chicago')
    Timestamp('2024-04-08 18:00:00')

    >>> to_utc_naive('2024-04-08T18:00:00+00:00')
    Timestamp('2024-04-08 18:00:00')
    """
    # Handle obspy UTCDateTime — always UTC
    try:
        from obspy import UTCDateTime as _UTCDateTime
        if isinstance(t, _UTCDateTime):
            return pd.Timestamp(t.datetime).replace(tzinfo=None)
    except ImportError:
        pass

    # Convert to pd.Timestamp (handles str, datetime, pd.Timestamp)
    ts = pd.Timestamp(t)

    if ts.tzinfo is not None:
        # Already tz-aware: convert to UTC and strip tz
        return ts.tz_convert('UTC').tz_localize(None)
    else:
        # Naive input
        if tz_str is not None:
            # Interpret as tz_str, convert to UTC, strip tz
            return ts.tz_localize(tz_str).tz_convert('UTC').tz_localize(None)
        # tz_str is None → treat as UTC already
        return ts



# Functions from src/acous_resp.py

def cosine_taper(x, p=0.05):
    """Apply a cosine (Tukey) taper to x.

    Parameters
    ----------
    x : ndarray
        Input data.
    p : float, optional
        Fraction of the data tapered at *each* end (0 < p < 0.5).
        p=0.05 tapers 5 % at each end (10 % total).

    Returns
    -------
    ndarray
        Tapered copy of x.
    """
    window = tukey(len(x), alpha=p)
    return x * window


def butter_bandpass_filter(x, freqmin, freqmax, fs, order=4, zerophase=True):
    """Butterworth bandpass filter.

    Parameters
    ----------
    x : array-like
        Input signal.
    freqmin : float
        Lower corner frequency (Hz).
    freqmax : float
        Upper corner frequency (Hz).
    fs : float
        Sampling rate (Hz).
    order : int, optional
        Filter order (default 4).
    zerophase : bool, optional
        If True use filtfilt (zero phase); else lfilter (causal).

    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [freqmin / nyq, freqmax / nyq], btype='band')
    if zerophase:
        return filtfilt(b, a, x)
    return lfilter(b, a, x)


def spectral_whiten(x, eps=1e-10):
    """Spectral whitening (unit-amplitude spectrum).

    Parameters
    ----------
    x : array-like, shape (..., N)
        Input time series.  Whitening is applied along the last axis.
    eps : float, optional
        Regularisation floor added to the amplitude spectrum.

    Returns
    -------
    ndarray
        Whitened time series (real part of inverse FFT).
    """
    x = np.asarray(x)
    X = fft(x, axis=-1)
    Xw = X / (np.abs(X) + eps)
    return ifft(Xw, axis=-1).real


def cross_correlate(x, y):
    """Cross-correlate two equal-length time series.

    Computes C(τ) = Σ y(t) x(t − τ)  (y correlated with x).

    Parameters
    ----------
    x, y : ndarray, shape (N,)
        Input signals of equal length.

    Returns
    -------
    lags : ndarray, shape (2N−1,)
        Lag indices (samples).
    xcor : ndarray, shape (2N−1,)
        Cross-correlation values.

    Raises
    ------
    ValueError
        If x and y have different lengths.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    n = len(x)
    lags = correlation_lags(n, n, mode="full")
    xcor = correlate(y, x, mode="full")
    return lags, xcor


def linear_stack(X):
    """Mean stack along axis 0.

    Parameters
    ----------
    X : ndarray, shape (M, N)

    Returns
    -------
    ndarray, shape (N,)
    """
    return np.nanmean(X, axis=0)


def phase_weighted_stack(X, power=2):
    """Phase-weighted stack along axis 0.

    Parameters
    ----------
    X : ndarray, shape (M, N)
    power : int or float, optional
        Coherence exponent (default 2).

    Returns
    -------
    ndarray, shape (N,)
    """
    analytic = hilbert(X, axis=-1)
    mask = ~np.all(np.abs(analytic) == 0, axis=1)
    analytic_m = analytic[mask]
    phase = analytic_m / np.abs(analytic_m)
    coherence = np.abs(np.nanmean(phase, axis=0)) ** power
    return np.nanmean(X, axis=0) * coherence


def estimate_impulse_response(acor, xcor, eps=1e-10):
    """Estimate the system impulse response from ACF and XCF.

    Uses spectral division:  H(ω) = XCF(ω) / (ACF(ω) + reg)

    Parameters
    ----------
    acor : ndarray, shape (N,)
        Auto-correlation of the source signal.
    xcor : ndarray, shape (N,)
        Cross-correlation of source and receiver signals.
    eps : float, optional
        Regularisation factor (relative to 99th percentile of |ACF(ω)|).

    Returns
    -------
    h : ndarray, shape (N,)
        Impulse response (time domain, zero-lag centred via fftshift).

    Raises
    ------
    ValueError
        If acor and xcor have different lengths.
    """
    if len(acor) != len(xcor):
        raise ValueError("acor and xcor must have the same length.")
    n = len(acor)
    A = fft(acor)
    C = fft(xcor)
    reg = np.percentile(np.abs(A), 99.0) * eps
    H = C / (A + reg)
    h = ifft(H, n=n).real
    return fftshift(h)


def estimate_vibroseis_response(X, Y, whiten=False, w_eps=1e-10,
                                stack_type='linear', pws_power=2.0,
                                ir_eps=1e-10):
    """Estimate the vibroseis impulse response from multiple source-receiver pairs.

    Parameters
    ----------
    X : array-like, shape (M, N)
        Source waveforms (M realisations of length N).
    Y : array-like, shape (M, N)
        Receiver waveforms.
    whiten : bool, optional
        Apply spectral whitening to each ACF/XCF before stacking.
    w_eps : float, optional
        Whitening regularisation (default 1e-10).
    stack_type : {'linear', 'pws'}
        Stacking method.
    pws_power : float, optional
        Phase-weighted stack coherence exponent (default 2.0).
    ir_eps : float, optional
        Impulse-response regularisation (default 1e-10).

    Returns
    -------
    lags : ndarray, shape (2N−1,)
        Lag indices (samples).
    acor_stack : ndarray, shape (2N−1,)
        Stacked auto-correlation.
    xcor_stack : ndarray, shape (2N−1,)
        Stacked cross-correlation.
    h : ndarray, shape (2N−1,)
        Estimated impulse response.
    """
    acors, xcors = [], []
    for x, y in zip(X, Y):
        lags, acor = cross_correlate(x, x)
        lags, xcor = cross_correlate(x, y)
        acors.append(acor)
        xcors.append(xcor)

    acors = np.array(acors)
    xcors = np.array(xcors)

    if whiten:
        acors = spectral_whiten(acors, eps=w_eps)
        xcors = spectral_whiten(xcors, eps=w_eps)

    if stack_type == 'linear':
        acor_stack = linear_stack(acors)
        xcor_stack = linear_stack(xcors)
    elif stack_type == 'pws':
        acor_stack = phase_weighted_stack(acors, power=pws_power)
        xcor_stack = phase_weighted_stack(xcors, power=pws_power)
    else:
        raise ValueError(f"Unknown stack_type '{stack_type}'. Use 'linear' or 'pws'.")

    h = estimate_impulse_response(acor_stack, xcor_stack, eps=ir_eps)
    return lags, acor_stack, xcor_stack, h


def _extract_trace_data(tr, t, n):
    """Extract n samples from an obspy Trace starting at time t.

    Pads with zeros if the requested window extends beyond the trace.

    Parameters
    ----------
    tr : obspy.Trace
    t : obspy.UTCDateTime
        Start time of the window to extract.
    n : int
        Number of samples to extract.

    Returns
    -------
    ndarray, shape (n,)
    """
    fs = tr.stats.sampling_rate
    x = np.zeros(n)

    idx_start = int(round((t - tr.stats.starttime) * fs))
    idx_end = idx_start + n

    tr_start = max(0, idx_start)
    tr_end = min(len(tr.data), idx_end)

    if tr_start >= tr_end:
        return x

    x_start = max(0, -idx_start)
    x_end = x_start + (tr_end - tr_start)
    x[x_start:x_end] = tr.data[tr_start:tr_end]
    return x


def preprocess_stream(st, t1, t2, npts_tol=100, freqmin=4.0, freqmax=20.0,
                      taper_pct=0.05):
    """Pre-process raw continuous data before response estimation.

    drop short traces → merge (interpolate gaps) → fill masked → linear
    detrend → trim/pad to [t1, t2] → cosine taper → zero-phase bandpass.
    This is the pre-processing used to build DB 4.

    Parameters
    ----------
    st : obspy.Stream
        Raw continuous data (e.g. from io_sapphire.get_waveforms).  Traces
        may be modified in place; pass ``st.copy()`` to keep the raw data.
    t1, t2 : obspy.UTCDateTime
        Output time span; traces are zero-padded if shorter.
    npts_tol : int, optional
        Traces with fewer samples are dropped before merging (default 100).
    freqmin, freqmax : float, optional
        Bandpass corners in Hz (default 4–20 Hz, 4 corners, zero-phase).
    taper_pct : float, optional
        Cosine taper fraction at each end (default 0.05).

    Returns
    -------
    obspy.Stream
    """
    st = Stream([tr for tr in st if tr.stats.npts >= npts_tol])
    st.merge(method=1, fill_value='interpolate')    # interpolate gaps — no zero-fill jumps
    for tr in st:                                   # fill any remaining masked values
        if np.ma.is_masked(tr.data):                # (e.g. edge gaps too wide to interpolate)
            tr.data = np.ma.filled(tr.data, fill_value=0)
    st.detrend(type='linear')
    st.trim(t1, t2, pad=True, fill_value=0)
    st.taper(max_percentage=taper_pct, type='cosine')
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax,
              corners=4, zerophase=True)
    return st


def stream_vibroseis_response(st, source, receiver, intvs, buf=10.0,
                              evt0=None,
                              evlo=None, evla=None, evdp=0.0,
                              **kwargs):
    """Estimate the vibroseis response directly from an obspy Stream.

    Extracts source and receiver waveforms for each event interval,
    stacks the auto- and cross-correlations, and returns the impulse
    response as an obspy Stream.

    Parameters
    ----------
    st : obspy.Stream
        Continuous data for both source and receiver (already pre-processed:
        detrended, tapered, bandpass-filtered).
    source : str
        Deployment station code of the pilot/reference signal, e.g. 'S056a'.
    receiver : str
        Deployment station code of the receiver, e.g. 'S080a'.
    intvs : list of [starttime, endtime]
        Event intervals to stack.  Each element is [t_start, t_end] where
        times are obspy.UTCDateTime or any type accepted by obspy.
    buf : float, optional
        Buffer (s) prepended to each event window for tapered extraction
        (default 10.0).
    evt0 : obspy.UTCDateTime or None, optional
        Reference event time attached to output trace headers.  If None,
        defaults to the stream start time.
    evlo : float or None, optional
        Event longitude (degrees). Required for geometry headers.
    evla : float or None, optional
        Event latitude (degrees). Required for geometry headers.
    evdp : float, optional
        Event depth (km), default 0.0 (surface source).
    **kwargs
        Passed to :func:`estimate_vibroseis_response`:
        ``whiten``, ``w_eps``, ``stack_type``, ``pws_power``, ``ir_eps``.

    Returns
    -------
    obspy.Stream
        Three-trace stream:
        - ACR  (source auto-correlation stack)
        - XCR  (source-receiver cross-correlation stack)
        - IR   (impulse response)
        Each trace carries SAC headers ``b`` (start lag, s) and
        ``e`` (end lag, s), and ``starttime`` set relative to evt0.
        Returns an empty Stream if no valid intervals were found.
    """
    st_ = Stream()
    st_ += st.select(station=source).copy().merge(fill_value=0)
    st_ += st.select(station=receiver).copy().merge(fill_value=0)

    fs = st_[0].stats.sampling_rate
    dur = np.max([intv[1] - intv[0] for intv in intvs]) + 2 * buf
    n = int(round(dur * fs))

    half_samp = 0.5 / fs
    X, Y = [], []

    for intv in intvs:
        start_t = intv[0] - buf

        tr_x = st_.select(station=source)[0]
        if (tr_x.stats.starttime > start_t + half_samp or
                tr_x.stats.endtime < start_t + dur - half_samp):
            continue

        tr_y = st_.select(station=receiver)[0]
        if (tr_y.stats.starttime > start_t + half_samp or
                tr_y.stats.endtime < start_t + dur - half_samp):
            continue

        X.append(_extract_trace_data(tr_x, start_t, n))
        Y.append(_extract_trace_data(tr_y, start_t, n))

    X = np.array(X)
    Y = np.array(Y)

    if X.size == 0 or Y.size == 0 or X.shape != Y.shape:
        return Stream()

    lags, acor_stack, xcor_stack, h = estimate_vibroseis_response(X, Y, **kwargs)

    # Extract station coordinates from the receiver trace SAC headers
    _tr_rec = st_.select(station=receiver)[0]
    _sac    = getattr(_tr_rec.stats, 'sac', None)
    stlo    = getattr(_sac, 'stlo', None) if _sac else None
    stla    = getattr(_sac, 'stla', None) if _sac else None

    # Pre-compute geometry if all coordinates are provided
    _has_geom = all(v is not None for v in (evlo, evla, stlo, stla))
    if _has_geom:
        _dist_m, _az, _baz = gps2dist_azimuth(evla, evlo, stla, stlo)

    def _make_trace(data, station, channel, evt0):
        tr = Trace(data=data)
        tr.stats.sampling_rate = fs
        tr.stats.station = station
        tr.stats.channel = channel
        tr.stats.sac = AttribDict()
        tr.stats.sac.b = lags[0] / fs
        tr.stats.sac.e = lags[-1] / fs
        if evt0 is None:
            evt0 = tr.stats.starttime
        tr.stats.starttime = evt0 + tr.stats.sac.b
        tr.stats.sac.evt0  = evt0
        if _has_geom:
            tr.stats.sac.evlo = evlo
            tr.stats.sac.evla = evla
            tr.stats.sac.evdp = evdp
            tr.stats.sac.stlo = stlo
            tr.stats.sac.stla = stla
            tr.stats.sac.dist = _dist_m
            tr.stats.sac.az   = _az
            tr.stats.sac.baz  = _baz
        return tr

    return Stream([
        _make_trace(acor_stack, source,   "ACR", evt0=evt0),
        _make_trace(xcor_stack, receiver, "XCR", evt0=evt0),
        _make_trace(h,          receiver, "IR",  evt0=evt0),
    ])


def trim_response_lag(st_resp, evt0, lag_min=-12.0, lag_max=12.0):
    """Cut response traces to the lag window [lag_min, lag_max] (in place).

    Data are cast to float32 and the SAC ``b``/``e`` headers and
    ``starttime`` are updated.  This is the lag window stored in DB 4.
    Keep the window symmetric about zero lag: the picking-stage taper is
    applied to it, and an asymmetric window biases amplitude.

    Parameters
    ----------
    st_resp : obspy.Stream
        Output of :func:`stream_vibroseis_response`.
    evt0 : obspy.UTCDateTime
        Reference event time (zero lag).
    lag_min, lag_max : float, optional
        Lag window in seconds (default ±12 s).

    Returns
    -------
    obspy.Stream
        The same stream, trimmed.
    """
    for tr in st_resp:
        b  = tr.stats.sac.b
        fs = tr.stats.sampling_rate
        i1 = max(int(round((lag_min - b) * fs)), 0)
        i2 = min(int(round((lag_max - b) * fs)) + 1, len(tr.data))
        tr.data = tr.data[i1:i2].astype(np.float32)
        tr.stats.sac.b     = lag_min
        tr.stats.sac.e     = lag_max
        tr.stats.starttime = evt0 + lag_min
    return st_resp



# Functions from src/acous_pick.py

def prepare_stream(out):
    """Build the combined 4-trace stream from a get_resp_at() output dict.

    Keeps XCR and IR from the 'linear' combo (amplitude stack), renames
    XCR → WXCR and IR → WIR in the 'wpws' combo (onset stack), then
    concatenates into a single Stream with unique channel codes.

    Parameters
    ----------
    out : dict
        Dict returned by io_acous_resp.get_resp_at().  Must contain
        out['resp']['linear'] and out['resp']['wpws'] as obspy.Stream.

    Returns
    -------
    obspy.Stream
        Four traces: XCR, IR (linear), WXCR, WIR (wpws).
    """
    st_linear = out['resp']['linear'].copy()
    st_wpws   = out['resp']['wpws'].copy()

    # Keep only XCR and IR from the linear stack
    st_linear = Stream([tr for tr in st_linear
                        if tr.stats.channel in ('XCR', 'IR')])

    # Rename wpws channels: XCR → WXCR, IR → WIR
    for tr in st_wpws:
        tr.stats.channel = 'W' + tr.stats.channel   # XCR→WXCR, IR→WIR

    return st_linear + st_wpws


def _pick_prominent_peak(t, x, rel_height=0.5):
    """Pick the most prominent positive local peak in a raw (non-enveloped) trace.

    Unlike a plain height-based pick, prominence rejects a peak that is merely
    riding on a locally raised baseline (e.g. a broad low-frequency hump with a
    small ripple on top) in favor of a peak that genuinely stands out from its
    immediate surroundings.

    Parameters
    ----------
    t : ndarray
        Time array (s), uniformly sampled.
    x : ndarray
        Raw signal (not enveloped).
    rel_height : float, optional
        Relative height (fraction of prominence) at which the peak width is
        measured -- passed straight to scipy.signal.peak_widths. Default 0.5
        (half-max).

    Returns
    -------
    t_peak : float
        Time of the picked peak (s).
    hwhm : float
        Half-width at `rel_height` of the picked peak (s). np.nan if no local
        maximum was found (argmax fallback -- width is undefined for a
        boundary/monotone "peak").
    peak_height : float
        Signal value at the picked peak.
    """
    dt = t[1] - t[0]

    peaks, props = find_peaks(x, prominence=0, height=0)

    if len(peaks) == 0:
        peak_idx    = int(np.argmax(x))
        t_peak      = float(t[peak_idx])
        peak_height = float(x[peak_idx])
        hwhm        = np.nan
    else:
        peak_idx    = int(peaks[np.argmax(props['prominences'])])
        t_peak      = float(t[peak_idx])
        peak_height = float(x[peak_idx])
        widths, *_  = peak_widths(x, [peak_idx], rel_height=rel_height)
        hwhm        = 0.5 * float(widths[0]) * dt   # peak_widths returns full width -> halve it

    return t_peak, hwhm, peak_height


def pick_onset_band(st, c0, dc, fl, fh, cmin, cmax, fs_new,
                    k=0.02, tol_npts=10, taper_pct=0.05,
                    noise_t_min=-2.0, noise_t_max=-1.0,
                    source=None, n_intvs=None, resample=True, taper=True):
    """Pick onset travel time and amplitude within one narrow frequency sub-band.

    Fresh, self-contained picker -- does not call prepare_stream(), pick_onset(),
    _pick_arrival(), or _hwhm() above.  No Hilbert envelope: travel time is picked
    from the raw, band-filtered WXCR trace; amplitude from the raw IR trace in a
    small fixed window around that pick.

    Search-window sizing
    ---------------------
    Two independent half-widths around t0 = r/c0; the tighter one wins:

        w_half_dc = r * dc / (c0**2 - dc**2)
            Exact half-width of the celerity window [c0-dc, c0+dc] converted to
            time at range r = dist/c0.

        w_half_fc = 1 / (4*fc) + k / (fh - fl)
            Quarter-cycle-of-fc safety margin (keeps the search from crossing
            into a neighboring carrier lobe) plus a reciprocal-bandwidth margin
            (accounts for time-domain smearing from narrow-band filtering).
            fc = (fl+fh)/2.

        w_half = min(w_half_dc, w_half_fc)

    The window is centered on t0 and clamped to the hard global celerity bounds
    [cmin, cmax]:

        t_min = max(t0 - w_half, r/cmax)
        t_max = min(t0 + w_half, r/cmin)

    Parameters
    ----------
    st : obspy.Stream
        Stream containing WXCR and IR channels (e.g. from prepare_stream()),
        NOT yet resampled or band-filtered -- both are done here, on a copy.
        Geometry (dist) and event metadata are read from the WXCR SAC headers.
    c0 : float
        Predicted celerity (m/s).
    dc : float
        Half-width of the celerity search window (m/s) -- see w_half_dc above.
    fl, fh : float
        Sub-band frequency limits (Hz). fc = (fl+fh)/2, bw = fh-fl.
    cmin, cmax : float
        Hard global celerity bounds (m/s) -- physical limits independent of
        c0/dc, used only to clamp the final [t_min, t_max] window.
    fs_new : float
        Resampling rate (Hz). Used to convert the picked travel time to a
        sample index for the IR amplitude window (see tol_npts), and -- when
        resample=True -- also applied via Stream.interpolate() before picking.
        When resample=False, *st* must already be sampled at fs_new (the
        caller is responsible for this); pass its true sampling rate here.
    k : float, optional
        Reciprocal-bandwidth margin multiplier in w_half_fc. Default 0.02,
        chosen so w_half_fc stays under the hard 1/(2*fc) cycle-skip ceiling
        even for the highest sub-bands.
    tol_npts : int, optional
        Half-width (samples, at fs_new) of the IR amplitude search window
        around the picked WXCR travel time. Default 10 samples (10 ms at
        fs_new=1000 Hz) -- a phase-shift tolerance for spectral-division
        noise, not a cycle-skip constraint, so it is not scaled by band.
    taper_pct : float, optional
        Cosine taper max_percentage applied before band-filtering. Default
        0.05, matching the archive convention.
    noise_t_min, noise_t_max : float, optional
        Fixed IR noise window (s, relative to evt0) for SNR. Default
        [-2.0, -1.0]: 1 s wide, clear of both the zero-lag arrival and the
        taper/filter edge effects near the trace boundary.
    source : str or None, optional
        Source station code -- not stored in SAC headers, pass through for a
        complete CSV row.
    n_intvs : int or None, optional
        Number of stacked chirp intervals -- pass through for a complete CSV
        row.
    resample : bool, optional
        If True (default), resample *st* to fs_new here (via
        Stream.interpolate()) before tapering/filtering. Set False when the
        caller has already resampled *st* to fs_new once outside the loop --
        this step is band-independent, so repeating it per-band (e.g. once
        per entry of a BANDS list) is wasted work. The caller must guarantee
        *st* is truly at fs_new when passing resample=False.
    taper : bool, optional
        If True (default), apply the cosine taper here. Set False when the
        caller has already tapered *st* once outside the loop -- like
        resample, tapering does not depend on fl/fh, so it is also
        band-independent and safe to hoist out of a per-band loop.

    Returns
    -------
    dict
        Metadata: source, receiver, evt0_str, n_intvs, dist, az, baz.
        Band info: fl, fh, fc, bw.
        Window: w_half_dc, w_half_fc, w_half, t0, t_min, t_max.
        Pick: trav_time, trav_time_hwhm, c_est, amplitude, amp_snr.
        Pick fields are np.nan if the search window contains no samples.
    """
    fc = (fl + fh) / 2.0
    bw = fh - fl

    nan_pick = dict(
        trav_time=np.nan, trav_time_hwhm=np.nan,
        c_est=np.nan, amplitude=np.nan, amp_snr=np.nan,
    )

    # ── Resample (optional) + band-filter (fresh copy; don't mutate caller's stream) ──
    st = st.copy()
    if resample:
        st.interpolate(fs_new, method='lanczos', a=20)
    if taper:
        st.taper(max_percentage=taper_pct, type='cosine')
    st.filter('bandpass', freqmin=fl, freqmax=fh, corners=4, zerophase=True)

    tr_wxcr = st.select(channel='WXCR')
    tr_ir   = st.select(channel='IR')
    if not tr_wxcr or not tr_ir:
        return dict(source=source, receiver=None, evt0_str=None, n_intvs=n_intvs,
                    dist=np.nan, az=np.nan, baz=np.nan,
                    fl=fl, fh=fh, fc=fc, bw=bw,
                    w_half_dc=np.nan, w_half_fc=np.nan, w_half=np.nan,
                    t0=np.nan, t_min=np.nan, t_max=np.nan) | nan_pick
    tr_wxcr = tr_wxcr[0]
    tr_ir   = tr_ir[0]

    # ── Metadata from SAC headers ───────────────────────────────────────
    sac      = tr_wxcr.stats.sac
    evt0     = sac.evt0
    evt0_str = evt0.strftime('%Y%m%dT%H%M%S')

    meta = dict(
        source   = source,
        receiver = tr_wxcr.stats.station,
        evt0_str = evt0_str,
        n_intvs  = n_intvs,
        dist     = float(getattr(sac, 'dist', np.nan)),
        az       = float(getattr(sac, 'az',   np.nan)),
        baz      = float(getattr(sac, 'baz',  np.nan)),
    )

    r = meta['dist']
    t = tr_wxcr.times(type='relative', reftime=evt0)

    # ── Search-window sizing ────────────────────────────────────────────
    w_half_dc = r * dc / (c0**2 - dc**2)
    w_half_fc = 1.0 / (4.0 * fc) + k / bw
    w_half    = min(w_half_dc, w_half_fc)

    t0    = r / c0
    t_min = max(t0 - w_half, r / cmax)
    t_max = min(t0 + w_half, r / cmin)

    band_meta = dict(fl=fl, fh=fh, fc=fc, bw=bw,
                     w_half_dc=w_half_dc, w_half_fc=w_half_fc, w_half=w_half,
                     t0=t0, t_min=t_min, t_max=t_max)

    win_mask = (t >= t_min) & (t <= t_max)
    if not np.any(win_mask):
        return meta | band_meta | nan_pick

    t_wxcr_win = t[win_mask]
    x_wxcr_win = tr_wxcr.data[win_mask]

    # ── Travel time: most prominent raw peak in the windowed WXCR ───────
    tt_est, hwhm, _ = _pick_prominent_peak(t_wxcr_win, x_wxcr_win)
    c_est = r / tt_est if tt_est > 0 else np.nan

    # ── Amplitude: raw max in a small fixed window around tt_est on IR ──
    x_ir      = tr_ir.data
    idx_est   = int(round((tt_est - t[0]) * fs_new))
    idx_start = max(idx_est - tol_npts, 0)
    idx_end   = min(idx_est + tol_npts + 1, len(x_ir))
    x_ir_win  = x_ir[idx_start:idx_end]
    amplitude = float(np.max(x_ir_win)) if len(x_ir_win) else np.nan

    # ── SNR: fixed noise window before zero lag ─────────────────────────
    neg_mask = (t >= noise_t_min) & (t <= noise_t_max)
    if np.any(neg_mask) and np.any(x_ir[neg_mask] != 0):
        noise_rms = float(np.sqrt(np.mean(x_ir[neg_mask] ** 2)))
        amp_snr   = amplitude / noise_rms if noise_rms > 0 else np.nan
    else:
        amp_snr = np.nan

    return meta | band_meta | dict(
        trav_time      = tt_est,
        trav_time_hwhm = hwhm,
        c_est          = c_est,
        amplitude      = amplitude,
        amp_snr        = amp_snr,
    )



# Functions from src/acous_pred.py

def get_c_ad(T, gamma=1.4, R=287.058):
    """Adiabatic sound speed from temperature.

    Parameters
    ----------
    T     : float or array-like   Air temperature (K).
    gamma : float                 Specific heat ratio (dry air, default 1.4).
    R     : float                 Specific gas constant (J kg⁻¹ K⁻¹, default 287.058).

    Returns
    -------
    c_ad : float or array-like   Adiabatic sound speed (m/s).
    """
    return np.sqrt(gamma * R * T)


def get_wind_along_track(u, v, az):
    """Wind component projected along a propagation azimuth.

    Parameters
    ----------
    u  : float or array-like   East wind component (m/s).
    v  : float or array-like   North wind component (m/s).
    az : float                 Source-to-receiver azimuth, CW from North (°).

    Returns
    -------
    w_along : float or array-like   Along-track wind component (m/s).
    """
    az_rad  = np.radians(az)
    w_along = u * np.sin(az_rad) + v * np.cos(az_rad)
    return w_along


def get_c_eff(T, u, v, az, gamma=1.4, R=287.058):
    """Effective sound speed: c_ad(T) + wind component along propagation azimuth.

    Parameters
    ----------
    T     : float or array-like   Temperature (K).
    u     : float or array-like   East wind component (m/s).
    v     : float or array-like   North wind component (m/s).
    az    : float                 Propagation azimuth, CW from North (°).
    gamma : float                 Specific heat ratio (default 1.4).
    R     : float                 Specific gas constant (default 287.058 J kg⁻¹ K⁻¹).

    Returns
    -------
    c_eff : float or array-like   Effective sound speed (m/s).
    """
    c_ad   = get_c_ad(T, gamma=gamma, R=R)
    w_along = get_wind_along_track(u, v, az)
    return c_ad + w_along


def get_sound_profile_gradient(z, c):
    """Linear least-squares gradient of a vertical profile.

    Parameters
    ----------
    z : array-like   Heights (m).
    c : array-like   Profile values (e.g. sound speed or c_eff) at each height.

    Returns
    -------
    a : float   Gradient (s⁻¹ when c is in m/s and z in m).
    """
    return float(np.polyfit(z, c, 1)[0])



# Functions from src/acous_grad.py

def celerity_vec(a, r, freq, c0_ec, c0_aw):
    """Predicted celerity for an array of gradients at a single range.

    Vectorised equivalent of src.acous_pred.predict_celerity. Branch is
    selected elementwise on sign(a): Pierce creeping-wave for a < 0,
    tau-p ducted group celerity for a >= 0. The AWS surface correction
    C_pred = c0_aw + (pred - c0_ec) is applied, exactly as in
    predict_celerity.

    Parameters
    ----------
    a     : ndarray  Effective sound-speed gradients (1/s).
    r     : float    Source-receiver range (m).
    freq  : float    Acoustic frequency (Hz).
    c0_ec : float    Profile surface sound speed (m/s), from ERA5.
    c0_aw : float    AWS surface sound speed (m/s).

    Returns
    -------
    ndarray, same shape as `a`.
    """
    a = np.asarray(a, dtype=float)
    pred = np.empty_like(a)

    up = a < 0.0
    dn = ~up

    # --- Upward refracting: shadow zone, Pierce (1981) creeping wave
    if up.any():
        pred[up] = c0_ec / (
            1.0 + K_pierce * np.abs(a[up]) ** (2.0 / 3.0) * freq ** (-2.0 / 3.0)
        )

    # --- Downward refracting: surface duct, tau-p group celerity
    if dn.any():
        th = np.arctan(a[dn] * r / (2.0 * c0_ec))
        val = np.full(th.shape, c0_ec, dtype=float)
        ok = th > 1e-13
        if ok.any():
            t_ok = th[ok]
            val[ok] = c0_ec * np.tan(t_ok) / np.log(
                (1.0 + np.sin(t_ok)) / np.cos(t_ok))
        pred[dn] = val

    return c0_aw + pred - c0_ec


def build_lookup(r_arr, a_tab, freq, c0_ec, c0_aw):
    """Per-receiver celerity lookup table.

    Returns
    -------
    ndarray, shape (N_receivers, len(a_tab))
    """
    lut = np.empty((len(r_arr), len(a_tab)), dtype=np.float64)
    for j, r in enumerate(r_arr):
        lut[j] = celerity_vec(a_tab, r, freq, c0_ec, c0_aw)
    return lut



# Functions from src/io_aws.py

def _wind_components(ws, wd):
    """Meteorological wind → Cartesian components.

    Parameters
    ----------
    ws : float or array-like  Wind speed (m/s).
    wd : float or array-like  Wind direction, coming from (° CW from N).

    Returns
    -------
    u : float or array-like  Zonal (East positive) component (m/s).
    v : float or array-like  Meridional (North positive) component (m/s).
    """
    wd_rad = np.deg2rad(wd)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v


def _load_aws():
    """Lazy-load and cache the AWS CSV.  Called internally; not part of public API."""
    global _df_aws
    if _df_aws is None:
        df = pd.read_csv(PATH_AWS_DATA)
        # time column is stored as UTC naive strings
        df['time'] = pd.to_datetime(df['time'])
        # Standardise column names
        df = df.rename(columns={
            'temperature_c':      'T_C',
            'wind_speed_m/s':     'ws_ms',
            'wind_direction_deg': 'wd_deg',
        })
        # Derived columns
        df['T_K'] = df['T_C'] + 273.15
        df['u_ms'], df['v_ms'] = _wind_components(df['ws_ms'], df['wd_deg'])
        # Select and order columns
        df = df[['time', 'T_C', 'T_K', 'ws_ms', 'wd_deg', 'u_ms', 'v_ms']].copy()
        df = df.reset_index(drop=True)
        _df_aws = df
    return _df_aws


def get_aws(starttime, endtime, tz_str=None):
    """Load AWS time series between starttime and endtime (inclusive).

    Parameters
    ----------
    starttime, endtime : str | datetime | pd.Timestamp | obspy.UTCDateTime
        Query window endpoints. Any timezone-aware or naive format.
    tz_str : str or None
        IANA timezone name for interpreting naive inputs (e.g. 'America/Chicago').
        None → treat naive input as UTC.

    Returns
    -------
    df_aw : pd.DataFrame
        Filtered AWS data. Columns: time, T_C, T_K, ws_ms, wd_deg, u_ms, v_ms.
        Index reset to 0-based.

    Examples
    --------
    >>> df_aw = get_aws('2024-04-08 12:00', '2024-04-08 20:00')

    >>> df_aw = get_aws('2024-04-08 07:00', '2024-04-08 15:00',
    ...                 tz_str='America/Chicago')
    """
    t0 = to_utc_naive(starttime, tz_str)
    t1 = to_utc_naive(endtime,   tz_str)
    df = _load_aws()
    mask = (df['time'] >= t0) & (df['time'] <= t1)
    return df.loc[mask].reset_index(drop=True)


def get_aws_at(t, tz_str=None, t_tol=60.0):
    """Return the AWS observation closest in time to t.

    Parameters
    ----------
    t : str | datetime | pd.Timestamp | obspy.UTCDateTime
        Query time.
    tz_str : str or None
        IANA timezone for naive inputs (None = UTC).
    t_tol : float or None
        Maximum allowed time difference in seconds between t and the
        closest observation.  Raises ValueError if exceeded.
        None disables the check (always return nearest).

    Returns
    -------
    t_closest : pd.Timestamp  Actual observation time (UTC naive).
    T_C       : float         Temperature (°C).
    T_K       : float         Temperature (K).
    ws_ms     : float         Wind speed (m/s).
    wd_deg    : float         Wind direction, from (° CW from N).
    u_ms      : float         Zonal wind component (m/s).
    v_ms      : float         Meridional wind component (m/s).

    Raises
    ------
    ValueError
        If the closest observation is more than t_tol seconds from t.

    Examples
    --------
    >>> t_c, T_C, T_K, ws, wd, u, v = get_aws_at('2024-04-08 18:42:00')

    >>> t_c, T_C, T_K, ws, wd, u, v = get_aws_at('2024-04-08 13:42:00',
    ...                                            tz_str='America/Chicago')

    >>> # Strict: raise if gap > 10 s
    >>> t_c, T_C, T_K, ws, wd, u, v = get_aws_at('2024-04-08 18:42:00',
    ...                                            t_tol=10.0)
    """
    t_utc = to_utc_naive(t, tz_str)
    df    = _load_aws()
    dt_s  = (df['time'] - t_utc).dt.total_seconds().abs()
    idx   = dt_s.argmin()

    if t_tol is not None and dt_s.iloc[idx] > t_tol:
        raise ValueError(
            f"Closest AWS observation is {dt_s.iloc[idx]:.1f} s from t "
            f"(tolerance {t_tol:.0f} s). "
            f"Closest time: {df['time'].iloc[idx]}."
        )

    row = df.iloc[idx]
    return (row['time'], float(row['T_C']), float(row['T_K']),
            float(row['ws_ms']), float(row['wd_deg']),
            float(row['u_ms']),  float(row['v_ms']))



def get_era5_gradients_at(t, tz_str=None):
    """Read 0–50 m least-squares slopes (s^-1) at the nearest UTC hour.

    a_t is the adiabatic sound-speed slope; a_u and a_v are eastward and
    northward wind slopes. These slopes do not reconstruct the full profiles.
    """
    t_hr = to_utc_naive(t, tz_str).round('h')
    gradients = pd.read_csv(PATH_ERA5_GRADIENTS, parse_dates=['t_utc']).set_index('t_utc')
    if t_hr not in gradients.index:
        raise ValueError(f'No ERA5 gradients for {t_hr} UTC.')
    return gradients.loc[t_hr]


# Standalone data access and notebook helpers.

def load_example():
    """Load the trusted bundled pickle: stream, UTC-naive event_time, events."""
    with (ROOT / 'example.pkl').open('rb') as handle:
        bundle = pickle.load(handle)
    assert isinstance(bundle['stream'], Stream)
    assert len(bundle['events']) == 1325
    return bundle


def local_time(values):
    """UTC-naive timestamps -> timezone-aware Dallas local timestamps."""
    if isinstance(values, pd.Series):
        return values.dt.tz_localize('UTC').dt.tz_convert(TZ)
    if isinstance(values, pd.DatetimeIndex):
        return values.tz_localize('UTC').tz_convert(TZ)
    return to_utc_naive(values).tz_localize('UTC').tz_convert(TZ)


def complete_chirp_intervals(stream, events, event_time, win=3600.0, buf=10.0):
    """Only use chirps with full buffered extraction within all three traces.

    One hour is exported exactly. Edge chirps without a complete buffer are
    excluded, rather than inventing samples outside the exported data.
    """
    t = to_utc_naive(event_time)
    a, b = t - pd.Timedelta(seconds=win/2), t + pd.Timedelta(seconds=win/2)
    frame = events[(events.begin_time <= b) & (events.end_time >= a)].copy()
    if frame.empty:
        raise ValueError('No chirps in the one-hour window')
    fs = stream[0].stats.sampling_rate
    duration = frame.duration.max() + 2*buf
    lower = max(tr.stats.starttime for tr in stream)
    upper = min(tr.stats.endtime for tr in stream)
    intervals = []
    for row in frame.itertuples():
        begin, end = UTCDateTime(row.begin_time.isoformat()), UTCDateTime(row.end_time.isoformat())
        start = begin - buf
        if start >= lower - 0.5/fs and start + duration <= upper + 0.5/fs:
            intervals.append([begin, end])
    if not intervals:
        raise ValueError('No chirps have complete buffered coverage')
    return intervals, len(frame)


def prepare_pick_stream(responses, event_time):
    """Match DB 4 precision/window and DB 5 resampling/taper convention."""
    evt0 = UTCDateTime(to_utc_naive(event_time).isoformat())
    trimmed = {}
    for key, stream in responses.items():
        trimmed[key] = trim_response_lag(stream.copy(), evt0, -12.0, 12.0)
    prepared = prepare_stream({'resp': trimmed})
    prepared = Stream([tr for tr in prepared if tr.stats.channel in ('WXCR', 'IR')])
    prepared.interpolate(1000.0, method='lanczos', a=20)
    prepared.taper(max_percentage=0.05, type='cosine')
    return prepared


def pick_response(prepared, event_time, n_intervals, fl=10.0, fh=12.0):
    """Pick using the original DB 5 temperature-plus-wind search-window speed."""
    _, _, T_K, _, _, u, v = get_aws_at(event_time, t_tol=None)
    az = float(prepared.select(channel='WXCR')[0].stats.sac.az)
    # This coefficient is the original DB 5 builder's convention.
    c0 = float(20.05*np.sqrt(T_K) + u*np.sin(np.deg2rad(az)) + v*np.cos(np.deg2rad(az)))
    picked = pick_onset_band(prepared, c0=c0, dc=10.0, fl=fl, fh=fh,
        cmin=330.0, cmax=355.0, fs_new=1000.0, k=0.02, tol_npts=10,
        taper_pct=0.05, noise_t_min=-2.0, noise_t_max=-1.0,
        source=SOURCE, n_intvs=n_intervals, resample=False, taper=False)
    picked['c0_search'] = c0
    return picked


def station_geometry():
    """Geometry from original source coordinates and QC-selected station table."""
    stations = pd.read_csv(DATA/'stations/sapphire_location.csv', dtype={'station_1': str})
    selection = pd.read_csv(DATA/'stations/station_selection.csv')
    stations = stations.merge(selection, on='station', how='left')
    stations = stations[stations['select'].fillna(0) == 1].copy()
    source = load_example()['events'].iloc[0]
    geom = [gps2dist_azimuth(source.latitude, source.longitude, row.latitude, row.longitude)
            for row in stations.itertuples()]
    stations[['dist','az','baz']] = np.array(geom)
    return stations.set_index('station')


_METRIC_CACHE = {}

def load_metrics(receiver, start=None, end=None, fl=10.0, fh=12.0):
    """Read bundled original raw_band CSV; apply original time-window QC."""
    if receiver not in _METRIC_CACHE:
        path = DATA/'acoustic_metrics'/f'S056a_{receiver}_S007a_w03600.csv'
        df = pd.read_csv(path)
        df['evt0'] = pd.to_datetime(df['evt0_str'], format='%Y%m%dT%H%M%S')
        exclusions = pd.read_csv(DATA/'stations/station_time_exclusions.csv',
                                  parse_dates=['exclude_start','exclude_end'])
        for row in exclusions[exclusions.station == receiver].itertuples():
            df = df[~df.evt0.between(row.exclude_start,row.exclude_end)]
        _METRIC_CACHE[receiver] = df
    df = _METRIC_CACHE[receiver]
    df = df[(df.fl == fl) & (df.fh == fh)]
    if start is not None: df = df[df.evt0 >= to_utc_naive(start)]
    if end is not None: df = df[df.evt0 <= to_utc_naive(end)]
    return df.sort_values('evt0').reset_index(drop=True).copy()


def landmarks():
    frame = pd.read_csv(DATA/'experiment_timing.csv')
    labels = {'totality':'Eclipse', 'sunset':'Sunset', 'thunderstorm':'Thunderstorm'}
    return {labels[r.event_key]: pd.Timestamp(r.time).tz_localize(r.timezone)
            for r in frame.itertuples() if r.event_key in labels}


def circular_mean(azimuths):
    rad = np.deg2rad(np.asarray(azimuths,float))
    return float(np.degrees(np.arctan2(np.sin(rad).mean(),np.cos(rad).mean())) % 360)


def time_axes(axes, mark=True):
    for ax in np.asarray(axes).flat:
        ax.set_xlim(local_time(START)-pd.Timedelta(minutes=30),local_time(END)+pd.Timedelta(minutes=5))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2,tz=TZ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%I %p',tz=TZ))
        ax.grid(ls=':',alpha=0.25)
        if mark:
            for label,t in landmarks().items(): ax.axvline(t,color='0.35',ls='--',lw=0.8)


def minute_observations(receivers, times, geometry, min_intervals=20):
    """Match original centred 60 s bins, then retain minute-grid timestamps."""
    frames=[]
    for receiver in receivers:
        df = load_metrics(receiver, times[0]-pd.Timedelta(seconds=30), times[-1]+pd.Timedelta(seconds=30))
        df = df[df.n_intvs >= min_intervals].dropna(subset=['c_est'])
        df['t_utc'] = (df.evt0+pd.Timedelta(seconds=30)).dt.floor('1min')
        agg = df.groupby('t_utc').agg(C_o=('c_est','mean'),amp_snr=('amp_snr','mean'),n=('c_est','size')).reset_index()
        agg.insert(0,'receiver',receiver)
        frames.append(agg[agg.t_utc.isin(times)])
    out = pd.concat(frames,ignore_index=True)
    out['dist'] = out.receiver.map(geometry.dist)
    return out


def hourly_observations(receivers, times, geometry, min_intervals=20):
    """Original 300 s centred window mean; average SNR before its threshold."""
    frames=[]
    for receiver in receivers:
        df = load_metrics(receiver,times[0]-pd.Timedelta(seconds=150),times[-1]+pd.Timedelta(seconds=150))
        df = df[df.n_intvs >= min_intervals]
        for t in times:
            window=df[df.evt0.between(t-pd.Timedelta(seconds=150),t+pd.Timedelta(seconds=150))]
            if window.empty: continue
            frames.append(dict(receiver=receiver,t_utc=t,C_o=window.c_est.mean(),
                               amp_snr=window.amp_snr.mean(),n=len(window),dist=geometry.loc[receiver,'dist']))
    return pd.DataFrame(frames).dropna(subset=['C_o'])


def surface_atmosphere(times, tolerance=60.0):
    """AWS sample nearest each fit time, as in the original minute inversion."""
    aws=get_aws(times[0]-pd.Timedelta(minutes=5),times[-1]+pd.Timedelta(minutes=5)).sort_values('time')
    atmosphere=pd.merge_asof(pd.DataFrame({'t_utc':times}),aws[['time','T_K']],
        left_on='t_utc',right_on='time',direction='nearest',tolerance=pd.Timedelta(seconds=tolerance))
    atmosphere['c0_ad']=get_c_ad(atmosphere.T_K.to_numpy(float))
    return atmosphere.set_index('t_utc')


def fit_gradient_windows(observations, times, atmosphere, a_grid, min_snr=3.0, min_receivers=3, frequency=11.0):
    """Search both refraction regimes; return fitted gradients and RMS surface.

    SNR is thresholded AFTER averaging the window. The forward model uses AWS
    adiabatic sound speed for both surface-speed arguments; ERA5 is not fitted.
    """
    observations=observations[observations.amp_snr >= min_snr]
    groups={t:d for t,d in observations.groupby('t_utc')}
    surface=np.full((len(a_grid),len(times)),np.nan)
    rows=[]
    for k,t in enumerate(times):
        c0=float(atmosphere.loc[t,'c0_ad'])
        d=groups.get(t)
        n=0 if d is None else len(d)
        row=dict(t_utc=t,a_est_C=np.nan,rms_min=np.nan,n_receivers=n,c0_ad=c0)
        if n >= min_receivers and np.isfinite(c0):
            lookup=build_lookup(d.dist.to_numpy(float),a_grid,frequency,c0,c0)
            rms=np.sqrt(np.mean((d.C_o.to_numpy(float)[:,None]-lookup)**2,axis=0))
            surface[:,k]=rms
            best=int(np.argmin(rms))
            row.update(a_est_C=float(a_grid[best]),rms_min=float(rms[best]))
        rows.append(row)
    fit=pd.DataFrame(rows)
    fit['t_local']=local_time(fit.t_utc).dt.tz_localize(None)
    return fit,surface


def era5_comparison(times, geometry, used, representatives):
    """ERA5 directional gradients and celerities for Figure 4 comparison."""
    rows=[]
    for t in times:
        gradient=get_era5_gradients_at(t)
        _,_,T_K,_,_,u,v=get_aws_at(t)
        c0=float(get_c_ad(T_K))
        for group,receivers in used.items():
            az=circular_mean(geometry.loc[receivers,'az'])
            a_group=gradient.a_t + gradient.a_u*np.sin(np.deg2rad(az)) + gradient.a_v*np.cos(np.deg2rad(az))
            receiver=representatives[group]
            az_rec=float(geometry.loc[receiver,'az'])
            a_rec=gradient.a_t + gradient.a_u*np.sin(np.deg2rad(az_rec)) + gradient.a_v*np.cos(np.deg2rad(az_rec))
            c_model=float(celerity_vec(np.array([a_rec]),float(geometry.loc[receiver,'dist']),11.0,c0,c0)[0])
            rows.append(dict(group=group,t_utc=t,a_E=a_group,a_E_receiver=a_rec,c0_ad=c0,C_model=c_model,
                             regime='up' if a_rec<0 else 'down'))
    return pd.DataFrame(rows)
