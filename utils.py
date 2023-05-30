import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import math
import matplotlib.gridspec as gridspec
from tqdm import tqdm

DATA_PATH = Path("data")
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
EXP1_IMAGE_SIZE = 500
SHAPES_IMAGE_SIZE = 500
EXP2_IMAGE_SIZE = 750
IMAGE_PATH = Path("images")
FIG_PATH = Path("figures")
CHOICE_LOCATIONS = {
    'bottom left': (182, 405), 'right': (398, 162), 'top left': (136, 125),
    315: (421, 674), 180: (551, 130), 270: (618, 550), 135: (328, 73),
    0: (196, 615), 45: (76, 416), 90: (129, 191), 225: (671, 329)
}


def get_angle(p1, p2, size):
    p1, p2 = map(np.array, [p1, p2])
    p1[1] = size - p1[1]
    p2[1] = size - p2[1]
    p1 -= size // 2
    p2 -= size // 2

    return np.degrees(np.arctan2(p2[1], p2[0])) - np.degrees(np.arctan2(p1[1], p1[0]))


def rotate(x, y, angle, width, height):
    angle = np.radians(angle)
    sin, cos = np.sin(angle), np.cos(angle)
    x -= width // 2
    y -= height // 2
    newx = cos * x - sin * y
    newy = sin * x + cos * y
    return newx + (width // 2), newy + (height // 2)


def plot_heatmap(x, y, size, image_path, ax, rotation_angle=0, title=None, ylabel=None):
    # y = size-y
    img = Image.open(image_path).resize((size, size))
    if rotation_angle:
        img = img.rotate(rotation_angle)
        x, y = rotate(x, y, rotation_angle, size, size)

    heatmap, _, _ = np.histogram2d(x, y, bins=(np.arange(size + 1), np.arange(size + 1)))
    og_heatmap = heatmap.copy().T
    heatmap /= heatmap.max()
    heatmap = heatmap.T

    heatmap = gaussian_filter(heatmap, 23)

    ax.imshow(img, extent=[0, size, 0, size], aspect='equal')
    ax.imshow(heatmap, extent=[0, size, 0, size], origin='lower', cmap='Reds',
              alpha=0.3 * (heatmap > (heatmap.max() * 0.01)))
    ax.scatter(x, y, marker='o', s=2, c="red", alpha=0.7)
    yy, xx = np.mgrid[0:size, 0:size]
    ax.contour(xx, yy, heatmap, extent=[0, 750, 0, 750], levels=10, vmin=0, cmap='Reds', linewidths=1)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return og_heatmap


def savefig(fig: plt.Figure, name, tight=True, si=False):
    savepath = FIG_PATH / ("si" if si else "main")
    if tight:
        fig.tight_layout()
    fig.savefig(savepath / f"{name}.png", dpi=500)
    plt.close(fig)


def distance_permutation_test_across_locations(df, n_perm=10000):
    np.random.seed(96)
    real_distance = df["dist"].mean()
    real_x = df["x"].to_numpy()
    real_y = df["y"].to_numpy()
    real_choice_x = df["pc_choice_x"].to_numpy()
    real_choice_y = df["pc_choice_y"].to_numpy()
    indices = np.arange(real_x.size)
    perm_x = np.zeros((real_x.size, n_perm), dtype=float)
    perm_y = np.zeros((real_y.size, n_perm), dtype=float)
    for i in tqdm(range(n_perm), desc="Permuting virtual player choice locations"):
        perm_idx = np.random.permutation(indices)
        perm_x[:, i] = real_choice_x[perm_idx]
        perm_y[:, i] = real_choice_y[perm_idx]
    perm_dists = np.sqrt(((real_x[:, None] - perm_x) ** 2) + ((real_y[:, None] - perm_y) ** 2)).mean(axis=0)
    return (real_distance < perm_dists).mean()
