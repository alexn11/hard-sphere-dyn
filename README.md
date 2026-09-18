Hard sphere dynamics simulator
==============================

Generates a diagram of the collisions.

Inspired by [**Long time derivation of Boltzmann equation from hard sphere dynamics**](https://arxiv.org/abs/2408.07818) by *Yu Deng, Zaher Hani and Xiao Ma*.

# Requirements

Works with `Python 3.12`, `numpy 1.26.4` and `matplotlib 3.6.3`, probably other versions too.

This also requires the [`dot`](https://graphviz.org/doc/info/command.html) command line tool in order to render the graph into a SVG image file.

# Usage

Run:
```bash
python hard-spheres.py
```
This firstly generates a number of positions & velocities that correspond to the initial conditions of the hard sphere system.
Then it runs a simulation with a collision-adapted time step:
  - a sphere moves along a straight line at a speed and direction dictated by its velocity;
  - when two spheres get close to each other, they elastically collide and their velocities are changed to the rule given (1.2) p.4 of the article;
  - the length of the time step is reduced to the interval to the next collision time when a collision is expected to occur before the next time step.

The positions of the spheres are shown in a plot which is refreshed at each time step (but not in linear time).

When the user closes the plot window, the simulation stops and a plot of the evolution of the total number of collisions with respect to time appears.

Finally, when that plot is closed, the graph representing the whole collision history of the simulation is generated and saved into a `.dot` file.
This file is then processed using the `dot` program which will produce a `SVG` image file.

The spheres evolve in a square 2-torus.

The parameters of the simulation and the paths of the output files are modifiable on the command line, for more details run:
```bash
python hard-spheres.py  --help
```


