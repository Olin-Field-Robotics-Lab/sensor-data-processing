import rasterio

def normalize_data(band_data):
    flat = band_data.flatten()
    norm = (flat - flat.min()) / (flat.max() - flat.min())
    return norm.reshape(band_data.shape)