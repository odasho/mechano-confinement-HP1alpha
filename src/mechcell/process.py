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
    distance,
    dotdict,
    find_coordinates,
    find_coordinates_arr,
    lmbda_dist_out_from_center,
    scale_coordinates,
)


@dataclass
class Results:
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
        return {k: v for k, v in asdict(self).items()}


class ProcessImage:
    def __init__(self, metadata, data_dir=None, show_viewer=False):

        self._check_data_dir_path(data_dir)

        if not isinstance(show_viewer, bool):
            raise TypeError("show_viewer must be provided as True or False")

        self._metadata = dotdict(metadata)
        self._data_dir = data_dir
        self._results = Results()
        self._show_viewer = show_viewer

    def __call__(self):
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

            # view image
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

            # view property

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
        if data_dir is None:
            raise ValueError("The path to the data directory with images must be provided.")

        if not isinstance(data_dir, Path):
            raise TypeError("The path to the data directory must be provided as a pathlib.Path object.")

    def _process_metadata(self):
        self._results.image_file = self._metadata.image_file
        self._results.cell_type = self._metadata.cell_type
        self._results.condition = self._metadata.condition

    def _load_image(self):
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
        # Apply Gaussian blur
        blurred = cle.gaussian_blur(stacks, None, sigma_x=1, sigma_y=1, sigma_z=0)  # apply gaussian blur

        # Thresholding HP1a condensates
        binary_HP1 = cle.greater_constant(blurred, None, self._metadata.HP1_threshold)

        # Apply eroded_otsu_labeling - default settings
        labeled_HP1 = cle.eroded_otsu_labeling(binary_HP1, None, 2, 2)
        labeled_HP1 = label(labeled_HP1)

        # Thresholding nucleus
        binary_DAPI_nucleus = cle.greater_constant(blurred, None, self._metadata.nucleus_threshold)

        # Labeling the nucleus
        labeled_DAPI_nucleus = cle.connected_components_labeling_box(binary_DAPI_nucleus)
        # Exclude some smaller labels if we get more than one label for nucleus
        labeled_DAPI_nucleus = cle.exclude_small_labels(labeled_DAPI_nucleus, None, 350000)
        labeled_nucleus = label(labeled_DAPI_nucleus)

        # Closing the nucleus to avoid large holes in the object
        labeled_nucleus_closed = cle.closing_labels(labeled_nucleus, None, self._metadata.nucleus_closed)
        labeled_nucleus_closed = label(labeled_nucleus_closed)

        return labeled_HP1, labeled_nucleus_closed

    def _extract_condensate_properties(self, labeled_image):
        labeled_image_sitk = sitk.GetImageFromArray(labeled_image.astype(int))
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(labeled_image_sitk)

        condensate_ids = shape_stats.GetLabels()
        voxels_condensates = [shape_stats.GetPhysicalSize(condensate_id) for condensate_id in condensate_ids]

        self._results.condensate_ids = condensate_ids
        self._results.tot_num_condensates = len(condensate_ids)
        self._results.voxels_condensates = voxels_condensates

    def _extract_nucleus_properties(self, labeled_image):
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
        # Nucleus coordinates
        nucleus_coord = find_coordinates_arr(labeled_nucleus_closed)
        physical_coord_nucleus = nucleus_coord * scaling_factors

        # Extract nucleus axes dimensions
        perc = 1
        rotocentered_physical_coord, x_length_nuc, y_length_nuc, z_length_nuc = axis_lengths(
            physical_coord_nucleus, perc
        )

        # Extract condensates axes dimensions
        coord_cond = find_coordinates(labeled_HP1, one_object=False)  # return res_lst from function

        n = 3  # divide the list of coord_cond
        res_all = [coord_cond[i : i + n] for i in range(0, len(coord_cond), n)]

        conds = []  # list with coordinates for all conds (each one is an element)

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
        radius_nuc = x_length_nuc / 2

        coords_inner = []

        for i in x_lengths_HP1:
            radius_HP1 = i / 2
            ratio = radius_HP1 / radius_nuc

            lmbda_inner = (
                1 - ratio
            )  # 1 defines the whole nucleus, subtracting the major axis ratio makes up the inner space within the nucleus for each condensate

            # Define labeled inner object based on this distance out from the nucleus centroid
            labeled_inner, coord_inner = lmbda_dist_out_from_center(
                lmbda_inner, nucleus_coord, centroid_nucleus, labeled_nucleus_closed
            )

            coords_inner.append(coord_inner)

        return coords_inner

    def _define_nucleus_periphery(self, nucleus_coord, centroid_nucleus, labeled_nucleus_closed, scaling_factors):
        lmbda_inner_nuc = 0.95  # Remaining nucleus thickness shell
        labeled_inner_nuc, coord_inner_nuc = lmbda_dist_out_from_center(
            lmbda_inner_nuc, nucleus_coord, centroid_nucleus, labeled_nucleus_closed
        )

        periphery = cle.binary_subtract(labeled_nucleus_closed, labeled_inner_nuc)  # periphery object
        periphery_coord_arr = np.array(find_coordinates(periphery))
        physical_periphery_arr = periphery_coord_arr * scaling_factors

        self._results.coord_inner_nuc = len(coord_inner_nuc)

        return periphery_coord_arr, physical_periphery_arr

    def _sample_random_points(self, coords_inner, scaling_factors):

        physical_random_points = []

        for i in coords_inner:
            # Sample 1 random point in this inner space for each condensate
            random_points_voxel = np.array(
                random.choices(i, k=1)
            )  # choose k number of random points, in this case 1, returns an array
            random_points_scaled = random_points_voxel * scaling_factors
            physical_random_points.append(random_points_scaled)

        physical_random_points_arr = np.array([sublist[0] for sublist in physical_random_points])

        return physical_random_points_arr

    def _radial_distances(self, physical_random_points_arr, physical_centroid_nucleus, physical_centroid_conds_arr):

        dist_rp_r = []  # to store radial distances between random points to nucleus center
        dist_cond_r = []  # to store radial distances between center of HP1a condensates to nucleus center

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

        min_dist_random_p = []  # to store minimum peripheral distances from random points
        min_dist_conds_p = []  # to store minimum peripheral distances from HP1a condensate centers

        # compute all distances from random point to all points on periphery
        for i in physical_random_points_arr:
            distances = []
            for j in physical_periphery_arr:
                dist = distance([i], [j])
                distances.append(dist)

            # Compute the minimum distance of all distances between random point and periphery points
            min_dist = min(distances)
            min_dist_random_p.append(min_dist)

        # compute all distances from HP1a centroids to all points on periphery
        for i in physical_centroid_conds_arr:
            distances = []
            for j in physical_periphery_arr:
                dist = distance([i], [j])
                distances.append(dist)

            # Compute the minimum distance of all distances between HP1a condensates and periphery points
            min_dist = min(distances)
            min_dist_conds_p.append(min_dist)

        self._results.p_minimal_distance_random_points = min_dist_random_p
        self._results.p_minimal_distance_condensates = min_dist_conds_p

    def _min_peripheral_distances(
        self, physical_random_points_arr, physical_centroid_conds_arr, physical_periphery_arr
    ):
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
