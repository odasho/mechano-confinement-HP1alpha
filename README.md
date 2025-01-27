# Supplementary code for the paper: Nuclear mechano-confinement induces geometry-dependent HP1α condensate alterations

[![DOI](https://zenodo.org/badge/784937400.svg)](https://doi.org/10.5281/zenodo.14747053)

This repository contains supplementary code for the paper
> Hovet, O., Nahali, N., Halaburkova, A., Haugen, L., Paulsen, J., Progida, C. 2024.
> Nuclear mechano-confinement induces geometry-dependent HP1α condensate alterations
>
> Preprint available here: [doi.org/10.21203/rs.3.rs-4175426/v1](https://www.researchsquare.com/article/rs-4175426/latest).

Citation and link to Nature Communications Biology article will be made available after publication.

## Abstract
Cells sense external physical cues through complex processes involving signaling pathways, cytoskeletal dynamics, and transcriptional regulation to coordinate a cellular response. A key emerging principle underlying such mechanoresponses is the interplay between nuclear morphology, chromatin organization, and the dynamic behavior of nuclear bodies such as HP1α condensates. Here, applying Airyscan super-resolution live cell imaging, we report a hitherto undescribed level of mechanoresponse triggered by cell confinement below their resting nuclear diameter, which elicits changes in the number, size and dynamics of HP1α nuclear condensates. Utilizing biophysical polymer models, we observe radial redistribution of HP1α condensates within the nucleus, influenced by changes in nuclear geometry. These insights shed new light on the complex relationship between external forces and changes in nuclear shape and chromatin organization in cell mechanoreception.

## Software

`environment.yml` lists the dependencies (including version numbers) of the `conda` environment used to run the code in this repository. Install the `conda` environment with the following command:

    $ conda env create --file environment.yml

Our `mechcell` Python package contains procedures and algorithms for:
* processing images
* extracting properties from labeled nuclei and condensates
* tracking condensates over time

To install `mechcell`, clone this repository, navigate to the root, and run:

    $ pip install .

Molecular dynamics simulations were run with [LAMMPS](https://www.lammps.org/) version [2023-08-02](https://github.com/lammps/lammps/tree/stable_2Aug2023_update3).


## Repository structure, data availability and reproducibility

This repository consists of the following directories:
* `analysis` contains:
    * scripts for polymer simulation
    * demos for image processing
    * tracking of HP1α condensates over time
    * supplemental analysis from imaging data
* `data` contains:
    * metadata used for HeLa and IMR90 cell processing and property extraction
    * data from live and immunofluorescence experiments on HeLa and IMR90 cells from cell confiner
    * data from tracking experiments of HeLa cells
    * data used for correlation analysis
    * data from confinement recovery experiments
    * data of strong chromatin foci under confinement
* `figures` contains scripts to reproduce the article figures.
* `molecular_simulation` contains `lammps` script for setting up the polymer confinement shell and example file.
* `src/mechcell` contains the `mechcell` package.

## Citation

Cite preprint as:

> Hovet et al. Nuclear mechano-confinement induces geometry-dependent HP1α condensate alterations, 08 April 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-4175426/v1]

How to cite the Nature Communications Biology publication will be added after the publication is finalized.

## Having issues
If you have any troubles please file an issue in the GitHub repository.

## License
MIT
