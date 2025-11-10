from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import random
from loguru import logger
from matplotlib.patches import Ellipse, Patch
from typing import Optional, Dict, Callable, Tuple, Sequence
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_random_rgb_with_actions(dataloader, num_samples=20, ncols=4, seed=None, figsize=(12, 8)):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    def _to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _to_rgb(img):
        img = _to_numpy(img)
        # channel-first -> channel-last
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = img.astype(float)
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
        return img

    def _first_two(a):
        a = _to_numpy(a).reshape(-1)
        return None if a.size < 2 else a[:2]

    samples = []

    # -------- Prefer direct sampling from dataset-like objects --------
    if hasattr(dataloader, "__len__") and hasattr(dataloader, "__getitem__") and not hasattr(dataloader, "dataset"):
        N = len(dataloader)
        if N == 0:
            raise ValueError("Dataset/buffer is empty.")
        for idx in random.sample(range(N), k=min(num_samples, N)):
            item = dataloader[idx]
            rgb = _to_rgb(item["camera"])
            act = _first_two(item["action"])
            if act is None:
                continue
            samples.append((rgb, act))
            if len(samples) >= num_samples:
                break

    else:
        # -------- Streaming over a DataLoader or generic iterable --------
        for batch in dataloader:
            if isinstance(batch, dict):
                cams = batch["camera"]; acts = batch["action"]
                cams_np = _to_numpy(cams); acts_np = _to_numpy(acts)

                # Case A: batched dict → camera is 4D (B, H, W, 3) or (B, 3, H, W)
                if cams_np.ndim == 4:
                    B = cams_np.shape[0]
                    for i in range(B):
                        rgb = _to_rgb(cams_np[i])
                        # action can be (B, 2) or (2,)
                        act = _first_two(acts_np[i] if acts_np.ndim >= 2 and acts_np.shape[0] == B else acts_np)
                        if act is None:
                            continue
                        samples.append((rgb, act))
                        if len(samples) >= num_samples:
                            break

                # Case B: single-sample dict → camera is 3D (H, W, 3) or (3, H, W)
                elif cams_np.ndim == 3:
                    rgb = _to_rgb(cams_np)
                    act = _first_two(acts_np)
                    if act is not None:
                        samples.append((rgb, act))

                else:
                    # Unrecognized shape; skip
                    pass

            else:
                # Iterable of samples (each a dict)
                iterable = batch if hasattr(batch, "__iter__") else [batch]
                for item in iterable:
                    rgb = _to_rgb(item["camera"])
                    act = _first_two(item["action"])
                    if act is None:
                        continue
                    samples.append((rgb, act))
            if len(samples) >= num_samples:
                break

    if len(samples) == 0:
        raise ValueError("Could not collect any valid (camera, action) pairs to visualize.")

    # -------- Plot grid --------
    n = len(samples)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, (rgb, act) in zip(axes, samples):
        ax.imshow(rgb)
        ax.set_title(f"expert a = [{act[0]:.3f}, {act[1]:.3f}]", fontsize=10)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig

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

def _as_image(arr):
    """Return HxWx3 uint8 image from common formats."""
    arr = np.asarray(arr)
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    # Grayscale -> RGB
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1):
        arr = np.repeat(arr[..., :1], 3, axis=-1)
    # Normalize/clip to uint8
    if arr.dtype != np.uint8:
        vmin, vmax = np.min(arr), np.max(arr)
        if vmax > 1.0 or vmin < 0.0:
            arr = (arr - vmin) / (vmax - vmin + 1e-8)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return arr

def _get_item(buf, idx):
    """Handle list-like or dataset-like buffers that return dicts."""
    item = buf[idx]
    if not isinstance(item, dict) and isinstance(item, (list, tuple)) and len(item) > 0:
        item = item[0]
    return item

def _sample_indices(n, k, rng):
    """Sample min(k,n) unique indices (with replacement if n<k)."""
    if n >= k:
        return rng.choice(n, size=k, replace=False)
    else:
        return rng.choice(n, size=k, replace=True)

def visualize_buffer(buffer, name=None, num_samples=10, seed=42, line_width=80):
    """
    Visualize a single buffer: each row shows the camera image (left)
    and a text panel (right) with state and curvature arrays as strings.

    buffer: a dataset-like object whose items are dicts with keys:
            "camera" (HxWx3 or 3xHxW), "state" (np array), "curvature" (np array)
    name: optional title string
    num_samples: number of samples to draw
    seed: RNG seed for reproducibility
    line_width: max characters per line for array string formatting
    """
    n = len(buffer)
    rng = np.random.default_rng(seed)
    idxs = _sample_indices(n, num_samples, rng)

    # 2-column layout: image | text (one sample per row)
    fig, axes = plt.subplots(
        num_samples, 2,
        figsize=(10, 2.2 * num_samples),
        gridspec_kw={"width_ratios": [3, 2]},
        squeeze=False
    )
    if name is None:
        name = "Buffer"
    fig.suptitle(f"{name}: {num_samples} random samples", y=0.995, fontsize=14)

    for row, idx in enumerate(idxs):
        item = _get_item(buffer, int(idx))

        # --- Camera ---
        img = _as_image(item["camera"])
        ax_img = axes[row, 0]
        ax_img.imshow(img)
        ax_img.set_axis_off()
        ax_img.set_title(f"idx {int(idx)}", fontsize=9)

        # --- Text panel: state & curvature as strings ---
        ax_txt = axes[row, 1]
        ax_txt.set_axis_off()

        state = np.asarray(item["state"])
        curvature = np.asarray(item["curvature"])
        domain_label = np.asarray(item['domain_indicator'])

        # Pretty array strings (compact and wrapped)
        s_str = np.array2string(state, precision=4, threshold=20, max_line_width=line_width, suppress_small=True)
        c_str = np.array2string(curvature, precision=4, threshold=20, max_line_width=line_width, suppress_small=True)

        text = (
            f"state shape: {state.shape}\n"
            f"{s_str}\n\n"
            f"curvature shape: {curvature.shape}\n"
            f"{c_str}",
            f'domain_label: {domain_label}'
        )
        ax_txt.text(0, 1, text, va="top", ha="left", family="monospace", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# ---------- unchanged small utils ----------
def _to_TD(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x

def _align_pair(
    agent_actions: np.ndarray,
    expert_actions: np.ndarray,
    map_expert_to_agent: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    time_align: str = "min",
) -> Tuple[np.ndarray, np.ndarray]:
    A = _to_TD(np.asarray(agent_actions))
    E = _to_TD(np.asarray(expert_actions))
    if time_align == "min":
        T = min(A.shape[0], E.shape[0])
    elif time_align == "agent":
        T = A.shape[0]
    elif time_align == "expert":
        T = E.shape[0]
    else:
        raise ValueError("time_align must be {'min','agent','expert'}")
    A, E = A[:T], E[:T]
    if map_expert_to_agent is not None:
        E = map_expert_to_agent(E)
    if A.shape[1] != E.shape[1]:
        raise ValueError(
            f"Action dim mismatch after alignment: agent {A.shape[1]} vs expert {E.shape[1]}."
        )
    return A, E

def _chi2_radius(conf: float) -> float:
    """2D confidence radius r s.t. P(Chi2(df=2) <= r^2) = conf => r = sqrt(-2 ln(1-conf))."""
    conf = float(conf)
    if not (0.0 < conf < 1.0):
        raise ValueError("ellipse_conf must be in (0,1)")
    return np.sqrt(-2.0 * np.log(1.0 - conf))

def _add_axis_aligned_ellipse(ax, cx, cy, var_x, var_y, *, scale, edgecolor, lw=1.0, zorder=1):
    """Hollow (no fill) ellipse centered at (cx,cy), width=2*scale*sqrt(var_x), height=..."""
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(var_x) and np.isfinite(var_y)):
        return
    width  = 2.0 * scale * np.sqrt(max(var_x, 0.0))
    height = 2.0 * scale * np.sqrt(max(var_y, 0.0))
    if width == 0.0 and height == 0.0:
        return
    e = Ellipse((cx, cy), width=width, height=height, angle=0.0,
                facecolor='none', edgecolor=edgecolor, linewidth=lw, zorder=zorder)
    ax.add_patch(e)

def _add_quad_fit(ax, x_vals, y_vals):
    """Fit y = a x^2 + b x + c, plot as dotted black curve across data span."""
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return
    x, y = x[mask], y[mask]
    a, b, c = np.polyfit(x, y, deg=2)
    xs = np.linspace(np.min(x), np.max(x), 200)
    ys = a * xs**2 + b * xs + c
    ax.plot(xs, ys, linestyle=':', color='black', linewidth=1.5)

def _shade_kl_regions(ax, xmins, xmaxs, thresholds,
                      colors=('#e6f2ff', '#fff2cc', '#e6ffea'), alpha=0.35):
    """
    Shade 3 vertical regions along the x-axis using thresholds = [t_low_mid, t_mid_high].
    Returns legend handles (Patch objects) to build a single, shared legend.
    """
    if thresholds is None or len(thresholds) != 2:
        raise ValueError("Provide thresholds as an iterable of two floats: [low_mid, mid_high].")
    t1, t2 = float(min(thresholds)), float(max(thresholds))

    xmin = xmins
    xmax = xmaxs

    ax.axvspan(xmin, t1, facecolor=colors[0], alpha=alpha, ec='none', zorder=0)
    ax.axvspan(t1,  t2, facecolor=colors[1], alpha=alpha, ec='none', zorder=0)
    ax.axvspan(t2,  xmax, facecolor=colors[2], alpha=alpha, ec='none', zorder=0)

    # Legend handles (facecolor only; alpha ~ the same)
    handles = [
        Patch(facecolor=colors[0], edgecolor='none', alpha=alpha, label='Low KL'),
        Patch(facecolor=colors[1], edgecolor='none', alpha=alpha, label='Mid KL'),
        Patch(facecolor=colors[2], edgecolor='none', alpha=alpha, label='High KL'),
    ]
    return handles

# ---------- Modified plotting ----------
def plot_OPE_eval(
    data: dict,
    map_expert_to_agent_by_name: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
    *,
    loss: str = "mse",                 # {'mse','rmse','mae'}
    time_align: str = "min",           # {'min','agent','expert'}
    variance_type: str = "ellipse",    # {'ellipse','error_bar'}
    ellipse_conf: float = 0.68,        # used when variance_type='ellipse'
    kl_thresholds: Sequence[float] = (0.33, 0.66),  # [low_mid, mid_high] on x-axis (KL)
    figsize=(11, 5),
    savepath: Optional[str] = None,
):
    """
    Two subplots with either hollow ellipses or error bars (uniform color for all agents),
    shaded Low/Mid/High KL regions (global legend), and quadratic trend (dotted black).
    """
    variance_type = variance_type.lower()
    if variance_type not in {"ellipse", "error_bar"}:
        raise ValueError("variance_type must be either 'ellipse' or 'error_bar'.")

    # Collect stats per agent
    names, x_kl_mean, x_kl_var = [], [], []
    y_loss_mean, y_loss_var = [], []
    y_len_mean,  y_len_var  = [], []

    for agent, ev in data.items():
        mapper = None
        if map_expert_to_agent_by_name and agent in map_expert_to_agent_by_name:
            mapper = map_expert_to_agent_by_name[agent]

        kl_arr = np.asarray(ev.get("KL_divergence estimation", []), dtype=float)
        kl_mean = float(np.nanmean(kl_arr)) if kl_arr.size else np.nan
        kl_var  = float(np.nanvar(kl_arr, ddof=0)) if kl_arr.size else np.nan

        traj_losses, traj_lengths = [], []
        for Aa, Ee in zip(ev.get("agent_actions", []), ev.get("expert_actions", [])):
            A, E = _align_pair(Aa, Ee, map_expert_to_agent=mapper, time_align=time_align)
            T = A.shape[0]
            if T == 0:
                continue
            diff = A - E
            if loss == "mse":
                traj_loss = float(np.mean(diff**2))
            elif loss == "rmse":
                traj_loss = float(np.sqrt(np.mean(diff**2)))
            elif loss == "mae":
                traj_loss = float(np.mean(np.abs(diff)))
            else:
                raise ValueError("loss must be one of {'mse','rmse','mae'}")
            traj_losses.append(traj_loss)
            traj_lengths.append(T)

        if len(traj_losses) == 0:
            loss_mean = np.nan
            loss_var  = np.nan
            len_mean  = np.nan
            len_var   = np.nan
        else:
            loss_mean = float(np.mean(traj_losses))
            loss_var  = float(np.var(traj_losses, ddof=0))
            len_mean  = float(np.mean(traj_lengths))
            len_var   = float(np.var(traj_lengths, ddof=0))

        names.append(agent)
        x_kl_mean.append(kl_mean)
        x_kl_var.append(kl_var)
        y_loss_mean.append(loss_mean)
        y_loss_var.append(loss_var)
        y_len_mean.append(len_mean)
        y_len_var.append(len_var)

    # Single (uniform) color for all agent visuals
    point_color = 'chocolate'   # matplotlib default first color
    edge_color  = 'gray'
    error_color = 'gray'

    # Precompute stds for variance visuals
    x_std   = np.sqrt(np.maximum(x_kl_var, 0.0))
    loss_sd = np.sqrt(np.maximum(y_loss_var, 0.0))
    len_sd  = np.sqrt(np.maximum(y_len_var, 0.0))

    # Determine x-span for shading & trend
    finite_x = np.asarray(x_kl_mean, float)
    finite_mask = np.isfinite(finite_x)
    if np.any(finite_mask):
        xmin = float(np.nanmin(finite_x[finite_mask]))
        xmax = float(np.nanmax(finite_x[finite_mask]))
        margin = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        xmin -= margin
        xmax += margin
    else:
        xmin, xmax = 0.0, 1.0  # fallback

    # --- plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    # Shade KL regions on both axes + capture legend handles once
    region_handles = _shade_kl_regions(ax1, xmin, xmax, kl_thresholds)

    _shade_kl_regions(ax2, xmin, xmax, kl_thresholds)

    # ---------- Plot 1: Imitation loss vs KL ----------
    if variance_type == "ellipse":
        r = _chi2_radius(ellipse_conf)
        for cx, cy, sx, sy in zip(x_kl_mean, y_loss_mean, x_std, loss_sd):
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            _add_axis_aligned_ellipse(ax1, cx, cy, sx**2, sy**2, scale=r, edgecolor=edge_color, lw=1.0, zorder=1)
        ax1.scatter(x_kl_mean, y_loss_mean, s=0.8, c=point_color, zorder=2, marker = 'x')
    else:  # error_bar
        ax1.errorbar(x_kl_mean, y_loss_mean, xerr=x_std, yerr=loss_sd,
                     fmt='x', capsize=2.0, linestyle='none', ecolor=error_color, color=point_color, zorder=2, markersize = 6.0, elinewidth=0.8)

    ax1.set_xlim(xmin, xmax)
    # ax1.set_xlabel("Average KL divergence estimation") # no need to have xlabel here
    yl_label = {"mse":"Average imitation loss",
                "rmse":"Average imitation loss",
                "mae":"Average imitation loss"}[loss]
    ax1.set_ylabel(yl_label)
    ax1.grid(True, alpha=0.3)
    _add_quad_fit(ax1, x_kl_mean, y_loss_mean)

    # ---------- Plot 2: Trajectory length vs KL ----------
    if variance_type == "ellipse":
        r = _chi2_radius(ellipse_conf)
        for cx, cy, sx, sy in zip(x_kl_mean, y_len_mean, x_std, len_sd):
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            _add_axis_aligned_ellipse(ax2, cx, cy, sx**2, sy**2, scale=r, edgecolor=edge_color, lw=1.0, zorder=1)
        ax2.scatter(x_kl_mean, y_len_mean, s=0.8, c=point_color, zorder=2, marker = 'x')
    else:
        ax2.errorbar(x_kl_mean, y_len_mean, xerr=x_std, yerr=len_sd,
                     fmt='x', capsize=2.0, linestyle='none', ecolor=error_color, color=point_color, zorder=2, markersize = 6.0, elinewidth=0.8)

    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel("Average KL divergence estimation")
    ax2.set_ylabel("Average trajectory length")
    ax2.grid(True, alpha=0.3)
    _add_quad_fit(ax2, x_kl_mean, y_len_mean)

    # -------- Single, shared legend for KL bands --------
    fig.legend(handles=region_handles, loc = 'center right', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))

    # --- single, shared legend for KL bands ---
    fig_legend = fig.legend(
        handles=region_handles,
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)  # slightly above the axes
    )

    # reserve top margin so legend isn't clipped
    # fig.tight_layout(rect=[0, 0, 1, 0.92])   # <- key line
    
    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.show()
    return fig, (ax1, ax2)