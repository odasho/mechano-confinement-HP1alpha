#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import napari
import numpy as np
import pyclesperanto_prototype as cle
import SimpleITK as sitk
from aicsimageio.readers import CziReader
from numpy import linalg as LA
from radiomics import shape
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from .utils import (
    axis_lengths,
    dotdict,
    find_coordinates,
    find_coordinates_arr,
    lmbda_dist_out_from_center,
    scale_coordinates,
)


@dataclass
class Results:
    """Dataclass for storing results of the image processing."""

    image_file: str = field(init=False, repr=False)
    cell_type: str = field(init=False, repr=False)
    condition: str = field(init=False, repr=False)
    nucleus_id: int = field(init=False, repr=False)
    voxels_nucleus: float = field(init=False, repr=False)
    major_axis_nucleus: float = field(init=False, repr=False)
    intermediate_axis_nucleus: float = field(init=False, repr=False)
    minor_axis_nucleus: float = field(init=False, repr=False)
    sphericity: float = field(init=False, repr=False)
    elongation: float = field(init=False, repr=False)
    flatness: float = field(init=False, repr=False)
    coord_inner_nuc: float = field(init=False, repr=False)
    tot_num_condensates: int = field(init=False, repr=False)
    condensate_ids: list = field(init=False, repr=False)
    voxels_condensates: list = field(init=False, repr=False)
    major_axis_HP1: list = field(init=False, repr=False)
    intermediate_axis_HP1: list = field(init=False, repr=False)
    minor_axis_HP1: list = field(init=False, repr=False)
    r_dist_random_points: list = field(init=False, repr=False)
    r_dist_condensates: list = field(init=False, repr=False)
    p_minimal_distance_random_points: list = field(init=False, repr=False)
    p_minimal_distance_condensates: list = field(init=False, repr=False)
    physical_pixel_size_z: float = field(init=False, repr=False)
    physical_pixel_size_x: float = field(init=False, repr=False)
    physical_pixel_size_y: float = field(init=False, repr=False)
    pixel_size_um2: float = field(init=False, repr=False)
    voxel_size_um3: float = field(init=False, repr=False)

    def dict(self):
        """
        Convert the data stored in this Results object to a dictionary.

        Returns
        -------
        dict
            A dictionary containing all the attributes of the Results object.
        """
        return {k: v for k, v in asdict(self).items()}


class ProcessImage:
    """
    Process microscopy images and extract various properties.

    Calling an instance of this class will execute the microscopy image
    processing pipeline. Properties are extracted and displayed in a
    3D viewer using napari.

    Attributes
    ----------
    metadata : dict
        Metadata of the image containing parameters for processing.
    data_dir : Path
        Path to the directory containing the image.
    show_viewer : bool
        Flag indicating whether to display the results in napari viewer.

    Methods
    -------
    __call__():
        Execute the image processing pipeline.
    """

    def __init__(self, metadata, data_dir=None, show_viewer=False):
        """
        Parameters
        ----------
        metadata : dict
            Metadata of the image containing parameters for processing.
        data_dir : Path
            Path to the directory containing the image.
        show_viewer : bool, optional
            Flag indicating whether to display the results in napari viewer.
        """

        self._check_data_dir_path(data_dir)

        if not isinstance(show_viewer, bool):
            raise TypeError("show_viewer must be provided as True or False")

        self._metadata = dotdict(metadata)
        self._data_dir = data_dir
        self._results = Results()
        self._show_viewer = show_viewer

    def __call__(self):
        """
        Executes the complete image processing pipeline.

        Returns
        -------
        dict
            A dictionary with results of the image processing.
        """

        self._process_metadata()

        img_stack, scaling_factors = self._load_image()

        labeled_HP1, labeled_nucleus_closed = self._apply_filters_and_threshold(img_stack)

        self._extract_condensate_properties(labeled_HP1)

        self._extract_nucleus_properties(labeled_nucleus_closed)

        self._extract_axes_dimensions(labeled_HP1, labeled_nucleus_closed, scaling_factors)

        self._extract_centroid(labeled_HP1, labeled_nucleus_closed, scaling_factors)

        nucleus_coord, x_length_nuc, x_lengths_HP1 = self._extract_axes_dimensions(
            labeled_HP1, labeled_nucleus_closed, scaling_factors
        )
        centroid_nucleus, physical_centroid_nucleus, physical_centroid_conds_arr = self._extract_centroid(
            labeled_HP1, labeled_nucleus_closed, scaling_factors
        )
        self._geometric_space(x_length_nuc, x_lengths_HP1, nucleus_coord, centroid_nucleus, labeled_nucleus_closed)

        self._define_nucleus_periphery(nucleus_coord, centroid_nucleus, labeled_nucleus_closed, scaling_factors)

        periphery_coord_arr, physical_periphery_arr = self._define_nucleus_periphery(
            nucleus_coord, centroid_nucleus, labeled_nucleus_closed, scaling_factors
        )

        coords_inner = self._geometric_space(
            x_length_nuc, x_lengths_HP1, nucleus_coord, centroid_nucleus, labeled_nucleus_closed
        )
        self._sample_random_points(coords_inner, scaling_factors)

        physical_random_points_arr = self._sample_random_points(coords_inner, scaling_factors)

        self._radial_distances(physical_random_points_arr, physical_centroid_nucleus, physical_centroid_conds_arr)
        self._min_peripheral_distances(physical_random_points_arr, physical_centroid_conds_arr, physical_periphery_arr)

        coord_edge_p_random_points, coord_edge_p_cond = self._min_peripheral_distances(
            physical_random_points_arr, physical_centroid_conds_arr, physical_periphery_arr
        )

        if self._show_viewer:
            viewer = napari.Viewer()

            # View image
            viewer.add_image(
                img_stack,
                colormap="green",
                blending="additive",
                scale=(
                    self._results.physical_pixel_size_z,
                    self._results.physical_pixel_size_y,
                    self._results.physical_pixel_size_x,
                ),
                rendering="mip",
                name="Image_488",
                visible=True,
            )

            # View property
            viewer.add_labels(
                labeled_HP1,
                blending="additive",
                scale=(
                    self._results.physical_pixel_size_z,
                    self._results.physical_pixel_size_y,
                    self._results.physical_pixel_size_x,
                ),
                name="labeled_HP1",
                visible=True,
            )

            viewer.add_labels(
                labeled_nucleus_closed,
                blending="additive",
                scale=(
                    self._results.physical_pixel_size_z,
                    self._results.physical_pixel_size_y,
                    self._results.physical_pixel_size_x,
                ),
                name="labeled_nucleus_closed",
                visible=True,
            )

            viewer.add_points(
                periphery_coord_arr,
                blending="translucent",
                face_color="orange",
                size=1,
                scale=(
                    self._results.physical_pixel_size_z,
                    self._results.physical_pixel_size_y,
                    self._results.physical_pixel_size_x,
                ),
                name="periphery",
                visible=True,
            )

            viewer.add_points(
                physical_random_points_arr,
                blending="translucent",
                face_color="blue",
                size=1,
                name="random_points",
                visible=True,
            )

            viewer.add_points(
                physical_centroid_nucleus,
                blending="translucent",
                face_color="yellow",
                size=1,
                name="centroid_nucleus",
                visible=True,
            )

            viewer.add_points(
                physical_centroid_conds_arr,
                blending="translucent",
                face_color="red",
                size=1,
                name="centroid_condensates",
                visible=True,
            )

            viewer.add_points(
                coord_edge_p_random_points,
                blending="translucent",
                face_color="green",
                size=1,
                name="peripheral_point_for_random_points",
                visible=True,
            )

            viewer.add_points(
                coord_edge_p_cond,
                blending="translucent",
                face_color="magenta",
                size=1,
                name="peripheral_point_for_condensates",
                visible=True,
            )

            viewer.dims.ndisplay = 3
            viewer.scale_bar.visible = True
            viewer.scale_bar.unit = "um"

        return self._results.dict()

    def _check_data_dir_path(self, data_dir):
        """
        Validates the path to the data directory.

        Parameters
        ----------
        data_dir : Path
            Path to the directory containing the image.

        Raises
        ------
        ValueError
            If the path to the data directory with images is not provided.
        TypeError
            If the data directory path is not a pathlib.Path object.
        """

        if data_dir is None:
            raise ValueError("The path to the data directory with images must be provided.")

        if not isinstance(data_dir, Path):
            raise TypeError("The path to the data directory must be provided as a pathlib.Path object.")

    def _process_metadata(self):
        """Processes metadata and stores it in the results object."""

        self._results.image_file = self._metadata.image_file
        self._results.cell_type = self._metadata.cell_type
        self._results.condition = self._metadata.condition

    def _load_image(self):
        """
        Loads the image and scales it according to provided metadata.

        Returns
        -------
        stacks : ndarray
            A stack of image data.
        scaling_factors : ndarray
            An array of scaling factors for each dimension.

        Raises
        ------
        FileNotFoundError
            If the specified image file does not exist.
        """

        image_file = self._data_dir / self._metadata.image_file

        if not image_file.exists():
            raise FileNotFoundError(
                f"Could not find '{image_file}'. Check if provided 'data_dir' and 'image_file' are correct."
            )

        reader = CziReader(image_file)

        # Write resolutions to results dataclass
        z_res = reader.physical_pixel_sizes.Z
        x_res = reader.physical_pixel_sizes.X
        y_res = reader.physical_pixel_sizes.Y

        scaling_factors = np.array([z_res, y_res, x_res])

        self._results.physical_pixel_size_z = z_res
        self._results.physical_pixel_size_x = x_res
        self._results.physical_pixel_size_y = y_res
        self._results.pixel_size_um2 = y_res * x_res
        self._results.voxel_size_um3 = z_res * y_res * x_res

        # Viewing the substack of the nucleus in question
        stacks = reader.data[
            0,
            0,
            self._metadata.channel,
            self._metadata.view_zmin : self._metadata.view_zmax,
            self._metadata.view_xmin : self._metadata.view_xmax,
            self._metadata.view_ymin : self._metadata.view_ymax,
        ]

        return stacks, scaling_factors

    def _apply_filters_and_threshold(self, stacks):
        """
        Applies gaussian blur and thresholds to the image stack.

        Parameters
        ----------
        stacks : ndarray
            A stack of image data.

        Returns
        -------
        labeled_HP1 : ndarray
            Labeled image with detected condensates.
        labeled_nucleus_closed : ndarray
            Labeled image with nucleus segmentation, closed to remove holes.
        """

        # Apply Gaussian blur
        blurred = cle.gaussian_blur(stacks, None, sigma_x=1, sigma_y=1, sigma_z=0)

        # Thresholding HP1a condensates
        binary_HP1 = cle.greater_constant(blurred, None, self._metadata.HP1_threshold)

        # Apply eroded_otsu_labeling - default settings
        labeled_HP1 = cle.eroded_otsu_labeling(binary_HP1, None, 2, 2)
        labeled_HP1 = label(labeled_HP1)

        # Thresholding nucleus
        binary_DAPI_nucleus = cle.greater_constant(blurred, None, self._metadata.nucleus_threshold)

        # Labeling the nucleus
        labeled_DAPI_nucleus = cle.connected_components_labeling_box(binary_DAPI_nucleus)
        # Exclude smaller labels to ensure one nucleus label
        labeled_DAPI_nucleus = cle.exclude_small_labels(labeled_DAPI_nucleus, None, 350000)
        labeled_nucleus = label(labeled_DAPI_nucleus)

        # Closing the nucleus to avoid large holes in the object
        labeled_nucleus_closed = cle.closing_labels(labeled_nucleus, None, self._metadata.nucleus_closed)
        labeled_nucleus_closed = label(labeled_nucleus_closed)

        return labeled_HP1, labeled_nucleus_closed

    def _extract_condensate_properties(self, labeled_image):
        """
        Extracts properties of labeled condensates.

        Parameters
        ----------
        labeled_image : ndarray
            Labeled image with detected condensates.
        """

        labeled_image_sitk = sitk.GetImageFromArray(labeled_image.astype(int))
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(labeled_image_sitk)

        condensate_ids = shape_stats.GetLabels()
        voxels_condensates = [shape_stats.GetPhysicalSize(condensate_id) for condensate_id in condensate_ids]

        self._results.condensate_ids = condensate_ids
        self._results.tot_num_condensates = len(condensate_ids)
        self._results.voxels_condensates = voxels_condensates

    def _extract_nucleus_properties(self, labeled_image):
        """
        Extracts properties of labeled nucleus.

        Parameters
        ----------
        labeled_image : ndarray
            Labeled image with the nucleus segmented.
        """

        labeled_image_sitk = sitk.GetImageFromArray(labeled_image.astype(int))
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(labeled_image_sitk)

        shapeFeatures = shape.RadiomicsShape(labeled_image_sitk, labeled_image_sitk)
        sphericity = shapeFeatures.getSphericityFeatureValue()

        nucleus_ids = shape_stats.GetLabels()
        nucleus_id = nucleus_ids[0]
        voxels_nucleus = shape_stats.GetPhysicalSize(nucleus_id)

        self._results.nucleus_id = nucleus_id
        self._results.voxels_nucleus = voxels_nucleus
        self._results.sphericity = sphericity

    def _extract_axes_dimensions(self, labeled_HP1, labeled_nucleus_closed, scaling_factors):
        """
        Extracts axial dimensions of nucleus and condensates.

        Parameters
        ----------
        labeled_HP1 : ndarray
            Labeled image with detected condensates.
        labeled_nucleus_closed : ndarray
            Labeled image with closed nucleus segmentation.
        scaling_factors : ndarray
            Scaling factors for each dimension.

        Returns
        -------
        nucleus_coord : ndarray
            Coordinates of the nucleus.
        x_length_nuc : float
            Length of the major axis of the nucleus.
        x_lengths_HP1 : list
            Lengths of the major axes of the condensates.
        """

        # Nucleus coordinates
        nucleus_coord = find_coordinates_arr(labeled_nucleus_closed)
        physical_coord_nucleus = nucleus_coord * scaling_factors

        # Extract nucleus axes dimensions
        perc = 1
        rotocentered_physical_coord, x_length_nuc, y_length_nuc, z_length_nuc = axis_lengths(
            physical_coord_nucleus, perc
        )

        # Extract condensates axes dimensions
        coord_cond = find_coordinates(labeled_HP1, one_object=False)

        # Divide the list of coord_cond
        n = 3
        res_all = [coord_cond[i : i + n] for i in range(0, len(coord_cond), n)]

        # List with coordinates for all conds (one condensate is an element in the list)
        conds = []

        for i in range(len(res_all)):
            res = res_all[i]
            z1 = res[0]
            x1 = res[1]
            y1 = res[2]
            zippet = list(zip(z1, x1, y1))
            conds.append(zippet)

        scaled_coords = [scale_coordinates(sublist, scaling_factors) for sublist in conds]

        rotocentered_physical_coords_HP1 = []
        x_lengths_HP1 = []
        y_lengths_HP1 = []
        z_lengths_HP1 = []

        for coords in scaled_coords:
            rotocentered_physical_coord, x_length, y_length, z_length = axis_lengths(np.array(coords), perc=0)

            # Append the results to the respective lists
            rotocentered_physical_coords_HP1.append(rotocentered_physical_coord)
            x_lengths_HP1.append(x_length)
            y_lengths_HP1.append(y_length)
            z_lengths_HP1.append(z_length)

        # Calculate elongation and flatness of nucleus
        elongation_nuc = 1 - np.sqrt(y_length_nuc / x_length_nuc)
        flatness_nuc = 1 - np.sqrt(z_length_nuc / x_length_nuc)

        self._results.minor_axis_nucleus = z_length_nuc
        self._results.major_axis_nucleus = x_length_nuc
        self._results.intermediate_axis_nucleus = y_length_nuc
        self._results.elongation = elongation_nuc
        self._results.flatness = flatness_nuc

        self._results.minor_axis_HP1 = z_lengths_HP1
        self._results.major_axis_HP1 = x_lengths_HP1
        self._results.intermediate_axis_HP1 = y_lengths_HP1

        return nucleus_coord, x_length_nuc, x_lengths_HP1

    def _extract_centroid(self, labeled_HP1, labeled_nucleus_closed, scaling_factors):
        """
        Extracts centroid of nucleus and condensates.

        Parameters
        ----------
        labeled_HP1 : ndarray
            Labeled image with detected condensates.
        labeled_nucleus_closed : ndarray
            Labeled image with closed nucleus segmentation.
        scaling_factors : ndarray
            Scaling factors for each dimension.

        Returns
        -------
        centroid_nucleus : ndarray
            Centroid of the nucleus.
        physical_centroid_nucleus : ndarray
            Physical centroid of the nucleus.
        physical_centroid_conds_arr : ndarray
            Physical centroids of the condensates.
        """

        # Nucleus centroid
        stats_nucleus = regionprops(labeled_nucleus_closed)
        centroid_nucleus = np.array([s.centroid for s in stats_nucleus])
        physical_centroid_nucleus = centroid_nucleus * scaling_factors  # get physical coordinates

        # HP1a centroids
        stats_conds = regionprops(labeled_HP1)
        centroid_conds_arr = np.array([s.centroid for s in stats_conds])
        physical_centroid_conds_arr = centroid_conds_arr * scaling_factors

        return centroid_nucleus, physical_centroid_nucleus, physical_centroid_conds_arr

    def _geometric_space(self, x_length_nuc, x_lengths_HP1, nucleus_coord, centroid_nucleus, labeled_nucleus_closed):
        """
        Defines the geometric space within the nucleus.

        Parameters
        ----------
        x_length_nuc : float
            Length of the major axis of the nucleus.
        x_lengths_HP1 : list
            Lengths of the major axes of the condensates.
        nucleus_coord : ndarray
            Coordinates of the nucleus.
        centroid_nucleus : ndarray
            Centroid of the nucleus.
        labeled_nucleus_closed : ndarray
            Labeled image with closed nucleus segmentation.

        Returns
        -------
        coords_inner : list
            Coordinates inner to the nucleus for each condensate.
        """

        radius_nuc = x_length_nuc / 2

        coords_inner = []

        for i in x_lengths_HP1:
            radius_HP1 = i / 2
            ratio = radius_HP1 / radius_nuc

            # 1 defines the whole nucleus. Subtracting the major axis ratio
            # makes up the inner space within the nucleus for each condensate
            lmbda_inner = 1 - ratio

            # Define labeled inner object based on this distance out from the
            # nucleus centroid
            _, coord_inner = lmbda_dist_out_from_center(
                lmbda_inner, nucleus_coord, centroid_nucleus, labeled_nucleus_closed
            )

            coords_inner.append(coord_inner)

        return coords_inner

    def _define_nucleus_periphery(self, nucleus_coord, centroid_nucleus, labeled_nucleus_closed, scaling_factors):
        """
        Determines the periphery of the nucleus.

        Parameters
        ----------
        nucleus_coord : ndarray
            Coordinates of the nucleus.
        centroid_nucleus : ndarray
            Centroid of the nucleus.
        labeled_nucleus_closed : ndarray
            Labeled image with closed nucleus segmentation.
        scaling_factors : ndarray
            Scaling factors for each dimension.

        Returns
        -------
        tuple
            Arrays containing the coordinates of the periphery and its physical representation.
        """

        lmbda_inner_nuc = 0.95  # Remaining nucleus thickness shell
        labeled_inner_nuc, coord_inner_nuc = lmbda_dist_out_from_center(
            lmbda_inner_nuc, nucleus_coord, centroid_nucleus, labeled_nucleus_closed
        )

        periphery = cle.binary_subtract(labeled_nucleus_closed, labeled_inner_nuc)  # Defined periphery object
        periphery_coord_arr = np.array(find_coordinates(periphery))
        physical_periphery_arr = periphery_coord_arr * scaling_factors

        self._results.coord_inner_nuc = len(coord_inner_nuc)

        return periphery_coord_arr, physical_periphery_arr

    def _sample_random_points(self, coords_inner, scaling_factors):
        """
        Randomly samples points within the inner space of the nucleus for each condensate.

        Parameters
        ----------
        coords_inner : list
            Coordinates of the inner space of the nucleus for each condensate.
        scaling_factors : ndarray
            Scaling factors for each dimension.

        Returns
        -------
        physical_random_points_arr : ndarray
            Physical coordinates of the randomly sampled points.
        """

        physical_random_points = []

        for i in coords_inner:
            # Sample 1 random point in this inner space for each condensate
            random_points_voxel = np.array(random.choices(i, k=1))
            random_points_scaled = random_points_voxel * scaling_factors
            physical_random_points.append(random_points_scaled)

        physical_random_points_arr = np.array([sublist[0] for sublist in physical_random_points])

        return physical_random_points_arr

    def _radial_distances(self, physical_random_points_arr, physical_centroid_nucleus, physical_centroid_conds_arr):
        """
        Computes radial distances of random points and condensates from the nucleus center.

        Parameters
        ----------
        physical_random_points_arr : ndarray
            Physical coordinates of the randomly sampled points.
        physical_centroid_nucleus : ndarray
            Physical centroid of the nucleus.
        physical_centroid_conds_arr : ndarray
            Physical centroids of the condensates.
        """

        dist_rp_r = []  # To store radial distances between random points to nucleus center
        dist_cond_r = []  # To store radial distances between center of HP1a condensates to nucleus center

        for i in physical_random_points_arr:
            dist_random_points = LA.norm([i] - physical_centroid_nucleus)  # Euclidian distance
            dist_rp_r.append(dist_random_points)

        for i in physical_centroid_conds_arr:
            dist = LA.norm([i] - physical_centroid_nucleus)  # Euclidian distance
            dist_cond_r.append(dist)

        self._results.r_dist_random_points = dist_rp_r
        self._results.r_dist_condensates = dist_cond_r

    def _min_peripheral_distances(
        self, physical_random_points_arr, physical_centroid_conds_arr, physical_periphery_arr
    ):
        """
        Computes minimal peripheral distances between points and the nucleus periphery.

        Parameters
        ----------
        physical_random_points_arr : ndarray
            Physical coordinates of the randomly sampled points.
        physical_centroid_conds_arr : ndarray
            Physical centroids of the condensates.
        physical_periphery_arr : ndarray
            Physical coordinates of the nucleus periphery.

        Returns
        -------
        coord_edge_p_random_points : ndarray
            Closest periphery coordinates to random points.
        coord_edge_p_cond : ndarray
            Closest periphery coordinates to condensates.
        """

        # Compute all distances from random points and HP1a centroids to all points on periphery
        distances_random_to_periphery = cdist(physical_random_points_arr, physical_periphery_arr)
        distances_centroid_to_periphery = cdist(physical_centroid_conds_arr, physical_periphery_arr)

        # Calculate minimum distances and corresponding indices
        min_dist_random_p = np.min(distances_random_to_periphery, axis=1)
        indexes_min_dist_random_p = np.argmin(distances_random_to_periphery, axis=1)

        min_dist_conds_p = np.min(distances_centroid_to_periphery, axis=1)
        indexes_min_dist_conds_p = np.argmin(distances_centroid_to_periphery, axis=1)

        # Retrieve the closest coordinates for random points and centroids
        coord_edge_p_random_points = physical_periphery_arr[indexes_min_dist_random_p]
        coord_edge_p_cond = physical_periphery_arr[indexes_min_dist_conds_p]

        # Update the results
        self._results.p_minimal_distance_random_points = min_dist_random_p.tolist()
        self._results.p_minimal_distance_condensates = min_dist_conds_p.tolist()

        return coord_edge_p_random_points, coord_edge_p_cond
