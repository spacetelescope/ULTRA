## ULTRA
ULTRA stands for ULtra stable Telescope Research and Analysis. 
This repository comprises analytical algorithms which is used to compute
wavefront tolerances for segmented primary mirror relevant to the scenario of exo-earth imaging
with future large space telescopes. It leverages the PASTIS algorithm, and a close-loop recursive algorithm to estimate 
the static and dynamic wavefront tolerances for segmented mirrors during a science observation.


### Quickstart
Pre-install `conda` and `git` in your machine.

- Navigate to the directory you want to clone the repository into: 
```bash
$ cd /User/<YourUser>/repos/
```
- Before cloning this repository, follow the installation instructions for PASTIS: https://github.com/spacetelescope/PASTIS

- Create a new `ultra` conda environment using the `environment.yml` file present inside PASTIS repository:
```bash
$ cd PASTIS
$ conda env create --name ultra --file environment.yml
```

- Activate the conda environment:
```bash
$ conda activate ultra
```

- Clone the `ULTRA` repository:
```bash
$ git clone https://github.com/spacetelescope/ULTRA.git
```

- Update the newly created `ultra` conda environment with `environment.yml` file inside `ULTRA` repository:  
```bash
$ cd ULTRA
$ conda env update --file environment.yml
```

- Install the `ULTRA` package into `ultra` environment in develop mode:
```bash
$ python setup.py develop
```

- Install the `PASTIS` package into `ultra` environment in develop mode:
```bash
$ cd PASTIS
$ python setup.py develop
```

### Contribution guidelines
- Create a new branch from develop. 
- Open a new pull request for bugs, enhancement or suggestion using the branch.
- Rebase and fix merge conflicts on your branch whenever there is an update on develop.
- Assign reviewer on your pull request and mark it ready to review.
