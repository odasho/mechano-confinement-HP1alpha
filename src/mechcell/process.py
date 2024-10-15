#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyclesperanto_prototype as cle
import SimpleITK as sitk
from aicsimageio.readers import CziReader
from skimage.measure import label

from .utils import dotdict


@dataclass
class Results:
    image_file: str = field(init=False, repr=False)
    cell_type: str = field(init=False, repr=False)
    condition: str = field(init=False, repr=False)
    # nucleus_id: int = field(init=False, repr=False)
    # voxel_nucleus: float = field(init=False, repr=False)
    condensate_ids: list = field(init=False, repr=False)
    tot_num_condensates: int = field(init=False, repr=False)
    voxel_condensates: list = field(init=False, repr=False)
    physical_pixel_size_z: float = field(init=False, repr=False)
    physical_pixel_size_x: float = field(init=False, repr=False)
    physical_pixel_size_y: float = field(init=False, repr=False)
    pixel_size_um2: float = field(init=False, repr=False)
    voxel_size_um3: float = field(init=False, repr=False)

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class ProcessImage:
    def __init__(self, metadata, data_dir=None):

        self._check_data_dir_path(data_dir)

        self._metadata = dotdict(metadata)
        self._data_dir = data_dir
        self._results = Results()

    def __call__(self):
        self._process_metadata()

        img_stack = self._load_image()

        labeled_HP1, labeled_nucleus_closed = self._apply_filters_and_threshold(img_stack)

        self._property_extraction(labeled_HP1)
        self._property_extraction(labeled_nucleus_closed)

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

        return stacks

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

    def _property_extraction(self, labeled_image):
        labeled_image_sitk = sitk.GetImageFromArray(labeled_image.astype(int))
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(labeled_image_sitk)

        condensate_ids = shape_stats.GetLabels()
        voxels_condensates = [shape_stats.GetPhysicalSize(condensate_id) for condensate_id in condensate_ids]

        self._results.condensate_ids = condensate_ids
        self._results.tot_num_condensates = len(condensate_ids)
        self._results.voxel_condensates = voxels_condensates
