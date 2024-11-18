#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyclesperanto_prototype as cle
import tifffile
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def resolution_voxels(file):
    """
    Extract voxel resolutions from .tif file.

    Example
    -------
    >>> resolution_voxels("path/to/image.tif")
    (1.0, 0.2, 0.2)

    Parameters
    ----------
    file : str
        Input .tif file

    Returns
    -------
    (z, y, x) : tuple
        Tuple of three floats representing the voxel resolutions of z, y, x in µm
    """

    with tifffile.TiffFile(file) as tiff:

        # Retrieve the tags/metadata of the first page
        image_metadata = tiff.imagej_metadata
        tags = tiff.pages[0].tags

        x_resolution = tags["XResolution"].value[0] / tags["XResolution"].value[1]
        y_resolution = tags["YResolution"].value[0] / tags["YResolution"].value[1]

        # Calculate the pixel size in µm
        x_resolution_physical = 1 / x_resolution
        y_resolution_physical = 1 / y_resolution

        if image_metadata is not None:
            z_resolution_physical = image_metadata.get("spacing", 1.0)
        else:
            # if no z voxel is available, use default voxel size
            z_resolution_physical = 1.0

    return z_resolution_physical, y_resolution_physical, x_resolution_physical


def find_coordinates_arr(labeled_object):
    """
    Extract coordinates (z, y, x) from a labeled image as an array.

    Example
    -------
    >>> find_coordinates_arr(labeled_object)
    [[11.58541092 10.60721458  5.96350313]
    [10.72082802  9.14077939 11.43819453]]

    Parameters
    ----------
    labeled_object : pyclesperanto_prototype._tier0._pycl.OCLArray
        Input labeled image

    Returns
    -------
    array
        Single array of [z, y, x] coordinates.
    """

    props = regionprops(labeled_object)
    coordinates = [prop.coords for prop in props]

    all_coords = np.concatenate(coordinates)

    return all_coords


def scale_coordinates(coords, scaling_factors):
    return [(scaling_factors[0] * z, scaling_factors[1] * y, scaling_factors[2] * x) for z, y, x in coords]


def find_coordinates(labeled_object, one_object=True):
    """
    Extract coordinates (z, y, x) of labeled image for one object or multiple labeled objects.
    If one_object is set to True, a list with list of arrays are returned.
    If one_object is set to False, a list containing arrays for z, y and x coordinates are returned for each labeled object.

    Example
    -------
    >>> find_coordinates(labeled_object, one_object=True)
    [(16, 243, 220), (16, 243, 221), (16, 243, 222), ...]

    >>> find_coordinates(labeled_object, one_object=False)
    [array([16, 16, 16, ...]), array([243, 243, 243, ...]), array([220, 221, 222, ...])]

    Parameters
    ----------
    labeled_object : np.ndarray (pyclesperanto_prototype._tier0._pycl.OCLArray)
        A 3D array representing a labeled image where each distinct object has a unique integer label.
    one_object : bool, optional, default=True
        If True, returns the coordinates as a list of tuples (z, y, x).
        If False, returns the coordinates for all objects as a list of three arrays, one each for z, y, and x coordinates.

    Returns
    -------
    list of tuples or list of arrays
        If one_object is True:
            Returns a list of tuples [(z, y, x), ...] representing the coordinates of the first labeled object.
        If one_object is False:
            Returns three lists [z, y, x], where each list is an array of the corresponding coordinates for all labeled objects.
    """
    _, counts = np.unique(labeled_object, return_counts=True)

    s1, s2, s3 = labeled_object.shape

    index = np.argsort(labeled_object, axis=None, kind="stable")

    # Generate the associated position
    i1 = np.arange(s1).repeat(s2 * s3)[index]
    i2 = np.tile(np.arange(s2), s1).repeat(s3)[index]
    i3 = np.tile(np.arange(s3), s1 * s2)[index]

    groupLabels, groupSizes = np.unique(labeled_object, return_counts=True)
    groupOffsets = np.concatenate(([0], counts.cumsum()))

    dict_labels_and_voxels_ = {}

    for i, label_i in enumerate(groupLabels):
        if label_i == 0:
            continue
        start, end = groupOffsets[i], groupOffsets[i] + groupSizes[i]
        index = (i1[start:end], i2[start:end], i3[start:end])
        dict_labels_and_voxels_[label_i] = [index]

    res_lst = []

    for i in dict_labels_and_voxels_.values():
        for j in i:
            for k in j:
                res_lst.append(k)

    if one_object:

        nr1, nr2, nr3 = res_lst[0], res_lst[1], res_lst[2]

        result_object = zip(nr1, nr2, nr3)
        result_object = list(result_object)

        return result_object

    else:
        return res_lst


def axis_lengths(physical_coord, perc):
    """
    Perform centralization and rototranslation on object with coordinates z, y, x.
    'perc' is referred to a given percentage to remove voxel irregularities.
    For whole objects, 'perc' is set to 0.

    Major, intermediate and minor axes are calculated on the input object as well
    as bounding box measurements.

    Example
    -------
    >>> axis_lengths(physical_coord, perc)
    Rotocentered coordinates: [[ 0.01757894  0.01698008 -0.9510412 ...]]
    Major axis: 0.6386933534043071
    Intermediate axis: 0.3432123342547064
    Minor axis: 0.8645829046504754
    Bounding box size along the x-axis: 0.6386933534043071
    Bounding box size along the y-axis: 0.3432123342547064
    Bounding box size along the z-axis: 0.8645829046504754

    Parameters
    ----------
    physical_coord : [[ z  y  x]] array
        Single array of z, y, x coordinates
    perc : float
        A percentage

    Returns
    -------
    rotocentered_physical_coord : [[z y x]] array
        Centralized and rototranslated coordinates around z, y, x axes
    x_length_adj : float
        Major axis adjusted diameter of object
    y_length_adj : float
        Intermediate axis adjusted diameter of object
    z_length_adj : float
        Minor axis adjusted diameter of object
    """
    # switch z and x to be on form: x, y, z
    physical_coord[:, [0, 2]] = physical_coord[:, [2, 0]]

    T_x = (np.min(physical_coord[:, 0]) + np.max(physical_coord[:, 0])) / 2
    T_y = (np.min(physical_coord[:, 1]) + np.max(physical_coord[:, 1])) / 2
    T_z = (np.min(physical_coord[:, 2]) + np.max(physical_coord[:, 2])) / 2

    T = np.array([T_x, T_y, T_z])

    centered_physical_coord = physical_coord - T

    pca = PCA(n_components=2)
    pca.fit(centered_physical_coord[:, 0:2])
    PCA(n_components=2)

    rotocentered_physical_coord = centered_physical_coord.copy()
    rotocentered_physical_coord[:, 0:2] = np.matmul(rotocentered_physical_coord[:, 0:2], np.transpose(pca.components_))

    x_length_adj = np.percentile(rotocentered_physical_coord[:, 0], 100 - perc) - np.percentile(
        rotocentered_physical_coord[:, 0], perc
    )
    y_length_adj = np.percentile(rotocentered_physical_coord[:, 1], 100 - perc) - np.percentile(
        rotocentered_physical_coord[:, 1], perc
    )
    z_length_adj = np.percentile(rotocentered_physical_coord[:, 2], 100 - perc) - np.percentile(
        rotocentered_physical_coord[:, 2], perc
    )

    return rotocentered_physical_coord, x_length_adj, y_length_adj, z_length_adj


def lmbda_dist_out_from_center(lmbda, coord_nuc, centroid_nuc, labeled_nucleus):
    """
    A distance lambda from nucleus centroid to define a labeled inner shell object from the total nucleus object.

    Parameters
    ----------
    lmbda : float
        Value between 0 and 1 to define distance out from nucleus centroid to make up an inner shell where 1 reaches the outer periphery.
    coord_nuc : [[ z  y  x]] array
        Single array of z, y, x coordinates of the nucleus
    centroid_nuc : [[ z  y  x]] array
        Single array of z, y, x coordinate of the nucleus center
    labeled_nucleus : np.ndarray (pyclesperanto_prototype._tier0._pycl.OCLArray)
        A 3D array representing a labeled image of the nucleus
    viewer : Napari viewer (class 'napari.viewer.Viewer')
        Viewer for n-dimensional image visualization

    Returns
    -------
    labeled : np.ndarray (pyclesperanto_prototype._tier0._pycl.OCLArray)
        A labeled inner shell object defined by distance lmbda out from nucleus centroid
    coord_layer : [[ z  y  x]] array
        Single array of z, y, x coordinates labeled inner shell object
    """
    # generate points lmbda distance out from centroid
    new_points = lmbda * coord_nuc + (1 - lmbda) * centroid_nuc
    # calculate the image shape based on the points
    image_shape = (np.max(new_points[:, 0]) + 1, np.max(new_points[:, 1]) + 1, np.max(new_points[:, 2]) + 1)
    # convert dimensions to integers
    image_shape_int = tuple(int(dim) for dim in image_shape)
    # index the array with integers instead of float and create a binary image with the specified shape
    binary_image = np.zeros(image_shape_int, dtype=bool)
    # convert coordinates to integers
    new_points_int = new_points.astype(int)
    # set points as "on" in the binary image
    binary_image[new_points_int[:, 0], new_points_int[:, 1], new_points_int[:, 2]] = True
    # label connected components in the binary image
    labeled = label(binary_image)
    # additional check to keep points in overlapping with labeled_nucleus and is one object
    labeled = cle.binary_and(labeled, labeled_nucleus)
    coord_layer = find_coordinates(labeled, one_object=True)

    return labeled, coord_layer


def distance(point1, point2):
    """
    Calculating the Euclidian distance between 2 points in 3D-space of z, x, y
    Input example:

    [(1.9019, 26.656, 27.097)],
    [(2.2477, 30.233, 36.162)]

    Output: their corresponding distance in um: 9.75
    """
    distance = np.sqrt(
        (point1[0][0] - point2[0][0]) ** 2 + (point1[0][1] - point2[0][1]) ** 2 + (point1[0][2] - point2[0][2]) ** 2
    )

    return distance


def append_to_column(dictionary, column_name, value):
    """
    Append a value to a list in the dictionary. If the column (key) does not exist, it is created with an empty list.

    Parameters
    ----------
    dictionary : dict
        The dictionary to append the value to. Keys represent columns and values are lists.
    column_name : str
        The key (column name) in the dictionary to append the value to. If the key does not exist it will be created.
    value : any
        The value to append to the list corresponding to the column name.

    Example
    -------
    >>> data = {}
    >>> append_to_column(data, "Condition", "Non-Confined")
    >>> print(data)
    {'Condition': ["Non-Confined"]}
    """
    # Use setdefault to create an empty list if the column does not exist and append the value
    dictionary.setdefault(column_name, []).append(value)
    dictionary.setdefault(column_name, []).append(value)
