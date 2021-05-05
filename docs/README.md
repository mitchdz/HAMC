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
# note: sometimes initial build fails for some reason. If that happens, run `CC=gcc cmake3 ../src/` one more time and then run `make hamc`.

# copy pbs job script to build directory
cp ../run_hamc.pbs ./
# !! make sure to modify run_hamc.pbs so the BUILD_DIR is correct for you.

# run the pbs script
qsub run_hamc.pbs
```
---
## Timing Analysis
### Testing specs
- CPU: Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz
- GPU: Nvidia GTX 980Ti

| Command | CPU time (min) | GPU time (min) | speedup |
| --- | --- | --- | --- |
| `./hamc -a test -n 2 -p 500 -t 10 -w 30 -s 10` | 0.01 | 0.01 | 0.97 |
| `./hamc -a test -n 2 -p 512 -t 10 -w 30 -s 10` | 0.01 | 0.01 | 1.04 |
| `./hamc -a test -n 2 -p 1024 -t 10 -w 30 -s 10` | 0.07 | 0.04 | 1.70 |
| `./hamc -a test -n 2 -p 2000 -t 10 -w 120 -s 10` | 0.57 | 0.22 | 2.64 |
| `./hamc -a test -n 2 -p 4800 -t 20 -w 60 -s 10` | 7.57 | 1.36 | 5.56 |
| `./hamc -a test -n 2 -p 6000 -t 20 -w 60 -s 10` | 14.78 | 2.27 | 6.52 |
| `./hamc -a test -n 2 -p 12000 -t 20 -w 60 -s 10` | 117.73 | 13.24 | 8.89 |
| `./hamc -a test -n 2 -p 24000 -t 20 -w 60 -s 10` | 938.88 | 70.46 | 13.32 |
| `./hamc -a test -n 2 -p 32771 -t 264 -w 274 -s 10` | N/A | N/A | N/A |


---
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
