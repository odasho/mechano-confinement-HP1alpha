#!/usr/bin/env python
# -*- coding: utf-8 -*-

import napari
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
import tifffile
from laptrack import LapTrack
from laptrack.data_conversion import convert_split_merge_df_to_napari_graph
from skimage.measure import label, regionprops_table


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
        Input .tif file.

    Returns
    -------
    (z, y, x) : tuple
        Tuple of three floats representing the voxel resolutions of z, y, x in µm.
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
            # If no z voxel is available, use default voxel size
            z_resolution_physical = 1.0

    return z_resolution_physical, y_resolution_physical, x_resolution_physical


def view_process_label_object(stack, threshold, y_res, x_res, nr_images):
    """
    Extract objects of interest from image stack visualized with Napari.

    Parameters
    ----------
    stack : ndarray
        Input stack file.
    threshold : int
        Threshold values to apply to stack images to extract objects of interest.
    y_res : float
        Physical resolution in the y-dimension used for scaling in Napari.
    x_res : float
        Physical resolution in the x-dimension used for scaling in Napari.
    nr_images : int
        Number of images in the image stack.

    Returns
    -------
    labels_per_frame : List
        A list of labeled objects for each frame in the stack.
    viewer : napari.Viewer
        A Napari viewer instance with the added original, blurred and labeled images.
    """

    # Launch napari viewer
    viewer = napari.Viewer()

    # Add stack to viewer
    viewer.add_image(stack, name="stack", scale=(y_res, x_res))

    # Initialize lists to store properties
    labels_per_frame = []

    for idx in range(nr_images):
        # Apply Gaussian blur to each image
        blurred_image = cle.gaussian_blur(stack[idx], None, sigma_x=1, sigma_y=1, sigma_z=0)

        # Add blurred images
        viewer.add_image(blurred_image, name="blurred", scale=(y_res, x_res), visible=False)

        # Threshold and label
        HP1a_greater_constant = threshold
        binary_cond = cle.greater_constant(blurred_image, None, HP1a_greater_constant)

        # Apply eroded otsu labeling
        labeled_cond = cle.eroded_otsu_labeling(binary_cond, None, 2, 2)
        labeled_cond = label(labeled_cond)

        # Add labeled condensates to viewer
        viewer.add_labels(labeled_cond, blending="additive", scale=(y_res, x_res), name="labeled_cond", visible=False)

        labels_per_frame.append(labeled_cond)

    return labels_per_frame, viewer


def tracking_object(stack, track_cost_cutoff, splitting_cost_cutoff, threshold, y_res, x_res, nr_images):
    """
    Track objects of interest from image stack visualized with Napari and extract properties of centroid and area.

    Parameters
    ----------
    stack : ndarray
        Input stack file.
    track_cost_cutoff : int
        Cost cutoff value for connecting tracking object movements across frames for LapTrack package.
    splitting_cost_cutoff : int
        Cost cutoff value for splitting events during tracking for LapTrack package.
    threshold : int
        Threshold values to apply to stack images to extract objects of interest.
    y_res : float
        Physical resolution in the y-dimension used for scaling in Napari.
    x_res : float
        Physical resolution in the x-dimension used for scaling in Napari.
    nr_images : int
        Number of images in the image stack.

    Returns
    -------
    track_df : pandas.DataFrame
        Dataframe containing tracking information including labels, centroids, frame number, area and track IDs.
    labels_per_frame : list
        A list of labeled objects for each frame in the stack.
    viewer : napari.Viewer
        A Napari viewer instance with the added original, blurred, labeled and tracked objects.
    """

    # Initialize list to store properties
    regionprops = []

    labels_per_frame, viewer = view_process_label_object(stack, threshold, y_res, x_res, nr_images)

    for frame, label_ in enumerate(labels_per_frame):
        df = pd.DataFrame(regionprops_table(label_, properties=["label", "centroid", "area"]))
        df["frame"] = frame
        regionprops.append(df)
    regionprops_df = pd.concat(regionprops)

    # Use LapTrack package to obtain centroid x and y over each frame
    lt = LapTrack(track_cost_cutoff=track_cost_cutoff, splitting_cost_cutoff=splitting_cost_cutoff)
    track_df, split_df, merge_df = lt.predict_dataframe(
        regionprops_df.copy(),
        coordinate_cols=["centroid-0", "centroid-1"],
    )

    # Reset index in the dataframe
    track_df = track_df.reset_index()

    # Add tracking graphs
    graph = convert_split_merge_df_to_napari_graph(split_df, merge_df)

    # Visualize separate tracks
    viewer.add_tracks(track_df[["track_id", "frame", "centroid-0", "centroid-1"]], graph=graph, scale=(y_res, x_res))

    # Add labels
    new_labels = np.zeros_like(labels_per_frame)

    track_df["index"] = track_df["index"] + 1

    for _, row in track_df.iterrows():
        frame = int(row["frame"])
        inds = labels_per_frame[frame] == row["index"]
        new_labels[frame][inds] = int(row["track_id"]) + 1

    # View separate labels
    viewer.add_labels(new_labels, scale=(y_res, x_res))

    track_df = track_df[["centroid-0", "centroid-1", "frame", "track_id", "index"]]

    track_df = track_df.rename(columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x", "area": "pixel_size"})

    return track_df, viewer, labels_per_frame
