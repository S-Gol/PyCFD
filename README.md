# PyCFD
 
PyCFD is a 2-D Navier Stokes Finite Element solver. It works on a square grid and uses Numba's JIT compiler to speed up computations. 

![Vortex flow](https://github.com/S-Gol/PyCFD/blob/main/Examples/VortexExample.gif)

# Features

1. Fully dynamic flow
2. Arbitrary boundary conditions
3. Arbitrary geometries
4. Custom fluid properties

# Usage

There are two ways to use PyCFD. 

The first is to create an instance of the `FluidSim` class. Once that's done, you can use it's `.timestep()` function to handle each timestep. No rendering will be performed, and you must manually add velocities every frame. 

The second is to create an instance of the `FluidsRenderer` class. You can perform setup and then use `renderer.startContinousRender()` to automatically render the flow using OpenCV. It features several rendering modes.

Further examples are available in the `Jupyter Examples.ipynb` file. 