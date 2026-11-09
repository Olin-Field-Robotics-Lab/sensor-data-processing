import h5py
import numpy as np
from matplotlib import pyplot as plt
from h5_helpers import *

data_path = "data/dataset.h5"

with h5py.File(data_path, "r") as data:
    # visualize images
    some_image = data["train"]["sentinel2"][june_timestamp][image_idx]
    plt.figure()
    plot_bands(some_image, band_names)
    plt.figure()
    plot_rgb(some_image, "A color image")
    plt.figure()
    plot_timestamps(data, image_idx)

    # visualize labels
    plt.figure()
    some_label = data["train"]["labels"]["field"][image_idx]

    x = np.argmax(some_label, axis=-1)
    binary_mask = np.sum(some_label, axis=-1)
    x[binary_mask == 0] = 6

    im = plt.imshow(x, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(
        im,
        fraction=0.035,
        pad=0.01,
        ticks=np.arange(0.5, 6.0),
        boundaries=np.arange(0, 7),
        extend="neither",
    )
    cbar.ax.set_yticklabels(legend)
    plt.title("field")
    plt.savefig("graphics/labels.png")
