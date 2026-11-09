import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def flatten_labels(labels):
    x = np.argmax(labels, axis=-1)
    binary_mask = np.sum(labels, axis=-1)
    x[binary_mask == 0] = 6
    return x

def plot_bands(some_image, band_names):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]
        im = ax.imshow(some_image[i], cmap="viridis")
        ax.set_title(band_names[i])
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(10, 12):
        axes[j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0.12)
    plt.savefig("graphics/bands.png")


def get_rgb(some_image):
    red = some_image[2]
    green = some_image[1]
    blue = some_image[0]
    rgb_image = np.dstack([red, green, blue])
    return rgb_image


def plot_rgb(some_image, title):
    rgb_image = get_rgb(some_image)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.savefig("graphics/rgb.png")


def plot_timestamps(data, image_idx):
    timestamp_keys = data["train"]["sentinel2"].keys()
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, timestamp in enumerate(timestamp_keys):
        ax = axes[i]
        some_image = data["train"]["sentinel2"][timestamp][image_idx]
        rgb_image = get_rgb(some_image)
        im = ax.imshow(rgb_image)
        ax.set_title(timestamp)
        ax.axis("off")

    for j in range(10, 12):
        axes[j].axis("off")

    plt.subplots_adjust(wspace=-0.1, hspace=0.12)
    plt.savefig("graphics/timestamps.png")


colors = ["red", "orange", "yellow", "lightgreen", "green", "black"]
legend = ["Very low", "Low", "Average", "High", "Very high", "Unlabeled"]

cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
norm = Normalize(vmin=0, vmax=5)

band_names = [
    "B02 (Blue 490nm)",
    "B03 (Green 560nm)",
    "B04 (Red 665nm)",
    "B05 (NIR 705nm)",
    "B06 (NIR 740nm)",
    "B07 (NIR 783nm)",
    "B08 (NIR 842nm)",
    "B8A (NIR 865nm)",
    "B11 (SWIR 1610nm)",
    "B12 (SWIR 2190nm)",
]

june_timestamp = "06_june_2019"
image_idx = 5555

crop_types = ["viticulture", "market", "field"]
