# Supplementary code for the paper: Nuclear mechano-confinement induces geometry-dependent HP1α condensate alterations

This repository contains supplementary code for the paper
> Hovet, O., Nahali, N., Halaburkova, A., Paulsen, J., Progida, C. 2024.
> Nuclear mechano-confinement induces geometry-dependent HP1α condensate alterations, preprint is under review at Nature Portfolio.
>
> Preprint available here: [doi.org/10.21203/rs.3.rs-4175426/v1](https://www.researchsquare.com/article/rs-4175426/latest).

This repository is a work in progress and will be finalized upon publication.

## Abstract
Cells sense external physical cues through complex processes involving signaling pathways, cytoskeletal dynamics, and transcriptional regulation to coordinate a cellular response. A key emerging principle underlying such mechanoresponses is the interplay between nuclear morphology, chromatin organization, and the dynamic behavior of nuclear bodies such as HP1α condensates. Here, applying super-resolution live cell imaging, we report a hitherto undescribed level of mechanoresponse triggered by cell confinement below their resting nuclear diameter, which elicits changes in the number, size and dynamics of HP1α nuclear condensates. Utilizing biophysical polymer models, we find that HP1α condensates become radially redistributed in the nucleus, dependent on the combined effect of HP1α's crosslinking activity and nuclear flattening. These insights shed new light on the complex relationship between external forces and changes in nuclear shape and chromatin organization in cell mechanoreception.

## Getting started

`environment_pinned.yml` lists the dependencies (including version numbers) of the `conda` environment used to run the code in this repository. Install the `conda` environment with the following command:

    $ conda env create --file environment_pinned.yml

Molecular dynamics simulations were run with [LAMMPS](https://www.lammps.org/) version [2023-08-02](https://github.com/lammps/lammps/tree/stable_2Aug2023_update3).

## Citation

Cite preprint as:

> Hovet et al. Nuclear mechano-confinement induces geometry-dependent HP1α condensate alterations, 08 April 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-4175426/v1]


## Having issues
If you have any troubles please file and issue in the GitHub repository.

## License
MIT
