from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def plot_scatter_line(data, xlabel, ylabel):
        """
        Plot scatter points and connect them with a line.

        Args:
            data (array-like): 1D array of values to plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        data = np.array(data)
        steps = np.arange(len(data))

        plt.figure(figsize=(8, 5))

        # Scatter + line
        plt.plot(steps, data, marker='o', linestyle='-', 
                color='steelblue', linewidth=2, markersize=6, 
                label=ylabel)

        # Labels and style
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{ylabel} vs {xlabel}", fontsize=14, weight='bold')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.show()

def speed_plot(agent_speeds, expert_speeds, collector_speeds, gridsize=50, title_prefix=""):
    """
    Visualize speed distributions for:
      1) Transferred actor (agent)
      2) Expert
      3) Target-domain data collector (PID)
    Each '..._speeds' is an iterable of [v_long, v_tran].
    """

    def _speed_mag(vx, vy):
        return np.sqrt(vx*vx + vy*vy)

    def _ecdf(x):
        x = np.sort(x)
        n = x.size
        y = np.arange(1, n+1) / n
        return x, y
    
    def _to_xy(arr):
        A = np.asarray(arr)
        return A[:,0], A[:,1]

    a_vx, a_vy = _to_xy(agent_speeds)
    e_vx, e_vy = _to_xy(expert_speeds)
    c_vx, c_vy = _to_xy(collector_speeds)

    plt.figure(figsize=(7,7))
    plt.scatter(a_vx, a_vy, alpha=0.4, label="Agent", s=10)
    plt.scatter(e_vx, e_vy, alpha=0.4, label="Expert", s=10)
    plt.scatter(c_vx, c_vy, alpha=0.4, label="Collector", s=10)
    plt.xlabel("v_long")
    plt.ylabel("v_tran")
    plt.title("Speed regime comparison in Target Domain")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    plt.show()

    # --- ECDF of speed magnitudes (all on one figure) ---
    plt.figure()
    a_m = _speed_mag(a_vx, a_vy)
    e_m = _speed_mag(e_vx, e_vy)
    c_m = _speed_mag(c_vx, c_vy)

    for data, label in [(a_m, "Agent"), (e_m, "Expert"), (c_m, "Collector")]:
        if data.size == 0:
            continue
        xs, ys = _ecdf(data)
        plt.plot(xs, ys, label=label)

    plt.xlabel("Speed magnitude")
    plt.ylabel("ECDF")
    plt.title((title_prefix + "Speed magnitude distribution (ECDF) in Target Domain").strip())
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.show()


def plot_speed_scatter(ax, arr, *, use_mean=False, cmap='viridis', add_colorbar=True, s=10):
    """
    Plot trajectory points colored by speed magnitude.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where the track map is already plotted.
    arr : list[np.ndarray] or np.ndarray
        arr[i][:, :2] -> (x, y) global positions at step i
        arr[i][:, 2:] -> (v_x, v_y) velocities at step i
    use_mean : bool
        If True, average across the first axis when arr[i] has multiple samples.
        If False, take the first row.
    cmap : str
        Colormap name (e.g. 'plasma', 'viridis', 'turbo').
    add_colorbar : bool
        Whether to add a colorbar.
    s : float
        Marker size for scatter points.
    """
    if isinstance(arr, np.ndarray):
        seq = list(arr)
    else:
        seq = list(arr)

    xy, vv = [], []
    for A in seq:
        A = np.asarray(A)
        if A.ndim == 1:
            x, y = A[0], A[1]
            vx, vy = A[2], A[3]
        else:
            slice_xy = A[:, :2]
            slice_vv = A[:, 2:4]
            if use_mean:
                x, y = np.nanmean(slice_xy, axis=0)
                vx, vy = np.nanmean(slice_vv, axis=0)
            else:
                x, y = slice_xy[0, 0], slice_xy[0, 1]
                vx, vy = slice_vv[0, 0], slice_vv[0, 1]
        xy.append((x, y))
        vv.append((vx, vy))

    xy = np.asarray(xy, dtype=float)
    vv = np.asarray(vv, dtype=float)
    mask = np.all(np.isfinite(xy), axis=1) & np.all(np.isfinite(vv), axis=1)
    xy, vv = xy[mask], vv[mask]
    if len(xy) == 0:
        return None

    speed = np.linalg.norm(vv, axis=1)
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=speed, s=s, cmap=cmap, edgecolors='none'
    )

    if add_colorbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Speed magnitude')

    return sc

def plot_action_diff_on_track(data_dict, ax, *, cmap='gray_r', s=14, alpha=0.95, add_colorbar=True, robust=False):
    """
    Plot GPS points on a given track axes, colored by the L2 difference between
    agent and expert actions. Darker = larger difference.

    Parameters
    ----------
    data_dict : dict
        Must contain:
          - 'gps' : np.ndarray of shape (N, 2)
          - 'agent_action' : np.ndarray of shape (N, 2)
          - 'expert_action' : np.ndarray of shape (N, 2)
    ax : matplotlib.axes.Axes
        The axes on which the track has already been plotted.
    cmap : str
        Colormap name. 'gray_r' makes darker points indicate larger difference.
    s : float
        Scatter point size.
    alpha : float
        Transparency of scatter points.
    add_colorbar : bool
        Whether to add a colorbar to the plot.
    robust : bool
        If True, trims color scale between 2nd–98th percentiles for robustness to outliers.

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        The scatter artist for the plotted points.
    norm : matplotlib.colors.Normalize
        The normalization object used for color scaling.
    """
    gps   = np.asarray(data_dict['gps'], dtype=float)
    a_act = np.asarray(data_dict['agent_action'], dtype=float)
    e_act = np.asarray(data_dict['expert_action'], dtype=float)

    # --- basic validation
    if gps.ndim != 2 or gps.shape[1] != 2:
        raise ValueError(f"gps must have shape (N, 2), got {gps.shape}")
    if a_act.shape != e_act.shape or a_act.shape[1] != 2 or a_act.shape[0] != gps.shape[0]:
        raise ValueError("agent_action and expert_action must both have shape (N, 2) matching gps.")

    # --- compute L2 differences
    diff = np.linalg.norm(a_act - e_act, axis=1)

    # --- filter out NaNs
    mask = np.all(np.isfinite(gps), axis=1) & np.isfinite(diff)
    gps, diff = gps[mask], diff[mask]
    if gps.size == 0:
        raise ValueError("No valid gps/action-diff data to plot.")

    # --- color normalization
    if robust:
        lo, hi = np.percentile(diff, [2, 98])
        if hi <= lo:
            lo, hi = np.min(diff), np.max(diff)
        norm = Normalize(vmin=lo, vmax=hi)
    else:
        norm = Normalize(vmin=np.min(diff), vmax=np.max(diff))

    # --- scatter plot
    sc = ax.scatter(
        gps[:, 0], gps[:, 1],
        c=diff, cmap=cmap, norm=norm,
        s=s, alpha=alpha, edgecolors='none'
    )

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Action Discrepancy (Agent vs Expert)')

    if add_colorbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(r'$||a_{\mathrm{agent}} - a_{\mathrm{expert}}||_2$')

    return sc, norm