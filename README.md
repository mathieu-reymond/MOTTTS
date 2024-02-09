How To
======

Requirements
------------

The project requires:

- CMake
- Compiler support for C++17 (so at least `g++-7`)
- Boost library
- The Eigen matrix library 3.3.

Instructions
------------

This project uses the `AI-Toolbox` library as a dependency. First run:
```
git submodule update --init --recursive
```


Then compile the project (please name the build folder `build` as the experiment
running/plotting code expects it like that).

```
mkdir build
cd build
cmake ..
make
cd ..
```

Running the baselines
---------------------

We have pre-generated 1Ok+ multi-objective multi-armed bandits (MOMABs). These are situated in the `generated_bandits` directory.

To run the baselines on any of these generated bandits, run:

```
./build/src/generate_mobandit_main -s 2 -a 10 -n 2 -e 10000 -p 100 -f generated_bandits/run_3_bandit_1.txt
```
where `s` stands for the starting seed, `n` the number of objectives, `a` the number of arms, `e` the number of experiments each baseline will execute on this MOMAB, `p` the number of particles used by particle filtering to estimate the utility function, and `f` the file of the generated bandit.

If no file exist at the provided path, then it will generated random (but challenging) MOMABs instead at the provided path.
