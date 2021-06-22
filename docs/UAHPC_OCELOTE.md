# running HAMC on UAHPC ocelete


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
