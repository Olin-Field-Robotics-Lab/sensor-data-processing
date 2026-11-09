import h5py
from h5_helpers import *

data_path = "data/dataset.h5"

with h5py.File(data_path, 'r') as data:
    some_image = data["train"]["sentinel2"][june_timestamp][image_idx]
    plot_bands(some_image, band_names)