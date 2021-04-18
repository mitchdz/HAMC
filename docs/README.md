# Hardware Accelerated McEliece Cryptosystem

CUDA based implementation of https://github.com/Varad0612/The-McEliece-Cryptosystem


## How to build and run:
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
CPU: Intel(R) Core(TM) i9-9900KF CPU @ 5.0GHz
GPU: GTX 980Ti

# Developers
* Mitchell Dzurick
* Mitchell Russel
* James Kuban
