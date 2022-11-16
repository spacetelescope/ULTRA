## ULTRA
ULTRA stands for ULtra stable Telescope Research and Analysis. 
This repository comprises analytical algorithms which is used to compute
wavefront tolerances of segmented primary mirror relevant to the scenario of exo-earth imaging
with future large telescope. It leverages the PASTIS algorithm, and a close-loop recursive algorithm to estimate 
the static and dynamic wavefront tolerances for segmented mirrors during a science observation.


### Quickstart
Pre-install `conda` and `git` in your machine.

- Navigate to the directory you want to clone the repository into: 
```bash
$ cd /User/<YourUser>/repos/
```

- Clone the repository:
```bash
$ git clone https://github.com/spacetelescope/ULTRA.git
```
- Navigate into the cloned `ULTRA` repository:  
```bash
$ cd ULTRA
```

- Create the `ultra` conda environment:  
```bash
$ conda env create --file environment.yml
```

- Activate the conda environment:
```bash
$ conda activate ultra
```

- Install the package into this environment in develop mode
```bash
$ python setup.py develop
```

### Contribution guidelines
- Create a new branch from develop. 
- Open a new pull request for bugs, enhancement or suggestion using the branch.
- Rebase your branch whenever there is any new update on develop.
- Assign reviewer on your pull request and mark it ready to review.
