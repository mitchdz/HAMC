# Hardware Accelerated McEliece Cryptosystem

CUDA based implementation of https://github.com/Varad0612/The-McEliece-Cryptosystem


## How to build and run on UAHPC ocelote:
```bash
# create build directory
mkdir -p build && cd build

# load CUDA modules
module load openmpi
module load cuda91/toolkit/9.1.85

# set up build directory
CC=gcc cmake3 ../src/

# build HAMC
make hamc

# copy pbs job script to build directory
cp ../run_hamc.pbs ./
# !! make sure to modify run_hamc.pbs so the BUILD_DIR is correct for you.

# run the pbs script
qsub run_hamc.pbs
```

## How to build and run (GENERIC):
First build the software:
```bash
$ mkdir -p build_dir && cd build_dir
$ cmake ../src/
$ make hamc
```

## GPU based execution:
```bash
$ ./hamc -a test
```

## CPU based execution:

```bash
$ ./hamc -a test -c
```
### Test computer specs:
* CPU: Intel(R) Core(TM) i9-9900KF CPU @ 5.0GHz
* GPU: GTX 980Ti

# Developers
* Mitchell Dzurick
* Mitchell Russel
* James Kuban
