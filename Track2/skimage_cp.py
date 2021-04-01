# Contest didn't have skimage package support, so
# I had to copy paste code from skimage github repository
# https://github.com/scikit-image/scikit-image


import numpy as np
from scipy import ndimage as ndi


class RegionProperties:
    def __init__(self, sl, label_image, label):
        self._label_image = label_image
        self._ndim = label_image.ndim
        self.label = label
        
        self.slice = sl
        self.image = self._label_image[self.slice] == self.label
        self.area = self.image.sum()
        self.centroid = tuple(self.get_cetroid(self.image).mean(axis=0))
    
    def get_cetroid(self, img):
        indices = np.nonzero(img)
        return np.vstack([indices[i] + self.slice[i].start
                          for i in range(self._ndim)]).T


def regionprops(img_label):
    regions = []

    objects = ndi.find_objects(img_label)
    
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1
        props = RegionProperties(sl, img_label, label)
        regions.append(props)
    
    return regions


def _resolve_neighborhood(selem, connectivity, ndim):
    if selem is None:
        if connectivity is None:
            connectivity = ndim
        selem = ndi.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        selem = np.asarray(selem, dtype=bool)
        # Must specify neighbors for all dimensions
        if selem.ndim != ndim:
            raise ValueError(
                "number of dimensions in image and structuring element do not"
                "match"
            )
        # Must only specify direct neighbors
        if any(s != 3 for s in selem.shape):
            raise ValueError("dimension size in structuring element is not 3")

    return selem


def label(image, background=None, return_num=False, connectivity=None):
    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    selem = _resolve_neighborhood(None, connectivity, image.ndim)
    result = ndi.label(image, structure=selem)

    if return_num:
        return result
    else:
        return result[0]
