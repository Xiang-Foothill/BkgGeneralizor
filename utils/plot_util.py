from matplotlib import pyplot as plt
import numpy as np


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