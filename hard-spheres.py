
__author__ = "Alexandre De Zotti"
__copyright__ = "Copyright (C) 2026 Alexandre De Zotti"
__license__ = "Public Domain"
__version__ = "1.0"

import argparse
from functools import partial
import subprocess
import time

import numpy as np
import matplotlib
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

k_Boltzmann = 1.381e-23
N_avogadro = 6.022e23

rng = np.random.default_rng()

class Maxwellian2D:
    def __init__(self, m, T, rng=None):
        self.m = m
        self.T = T
        self.rng = rng
        if(rng is None):
            self.rng = np.random.default_rng()
        self.compute_std()
    def compute_std(self):
        self.sigma = ((k_Boltzmann * self.T) / self.m)**0.5
    def sample(self, n):
        return self.rng.normal(loc=0., scale=self.sigma, size=[n, 2])

class CollisionFinder:
    def __init__(self, x: np.ndarray, v: np.ndarray, t: float, d2: float, eps: float = 0.):
        self.d2 = d2
        self.eps = eps
        self.update(x, v, t)
    def update(self, x: np.ndarray, v: np.ndarray, t: float):
        self.x = x
        self.v = v
        self.t = t
        self.invalid_collision_time = 2.*t + 1.
        self.compute_collisions()
    def compute_collisions(self):
        # that's quite inefficient, more than half the computations could be
        # note: there are ways to process symmetric matrices efficiently with numpy
        # also: not all these values will be needed/used
        self.dx = self.x[:, np.newaxis, :] - self.x[np.newaxis, :, :]
        self.dv = self.v[:, np.newaxis, :] - self.v[np.newaxis, :, :]
        self.dx2 = np.sum(self.dx**2, axis=-1)
        self.dv2 = np.sum(self.dv**2, axis=-1)
        self.dxdv = np.sum(self.dx*self.dv, axis=-1)
        reduced_discriminants = self.dxdv**2 - self.dv2 * (self.dx2 - self.d2)
        self.filter_discr = reduced_discriminants >= 0.
        self.collision_times = np.where(self.filter_discr,
                                   (- self.dxdv - reduced_discriminants**0.5) / self.dv2,
                                   self.invalid_collision_time)
        self.filter_t = (self.collision_times >= 0.) & (self.collision_times <= self.t)
        self.collision_times = np.where(self.filter_t,
                                        self.collision_times,
                                        self.invalid_collision_time)
    def get_next_collision(self) -> None | tuple[tuple[int, int], float]:
        i_1st_collision, j_1st_collision = np.unravel_index(np.argmin(self.collision_times), self.collision_times.shape)
        t_1st_collision = self.collision_times[i_1st_collision, j_1st_collision]
        if(t_1st_collision > self.t):
            return None
        return (i_1st_collision, j_1st_collision), t_1st_collision

class HardSphereDynamics:
    def __init__(self, sphere_r, base_dt, z0, box_size=0., eps=0.):
        self.r = sphere_r
        self.d = 2. * self.r
        self.d2 = self.d**2
        self.base_dt = base_dt
        self.x0: np.ndarray = z0[:,:,0]
        self.v0: np.ndarray = z0[:,:,1]
        self.box_size = box_size
        self.prepare_simulation()
        self.eps = eps
        self.collision_finder = None
        self.collisions = []
    def prepare_simulation(self):
        self.t = 0
        self.dt = self.base_dt
        self.x = self.x0.copy()
        self.v = self.v0.copy()
        self.N = len(self.x0)
    def linear_time_step(self):
        self.x += self.dt * self.v
        self.t += self.dt
    def prepare_step(self):
        self.dt = self.base_dt
        if(self.box_size > 0.):
            self.x = np.mod(self.x, self.box_size)
    def find_next_collision(self):
        if(self.collision_finder is None):
            self.collision_finder = CollisionFinder(x=self.x, v=self.v, t=self.dt, d2=self.d2, eps=self.eps)
        else:
            self.collision_finder.update(self.x, self.v, self.dt)
        next_collision = self.collision_finder.get_next_collision()
        if(next_collision is None):
            return None
        coll_idxs, coll_t = next_collision
        self.collisions.append([coll_idxs, coll_t + self.t])
        self.dt = coll_t
        return coll_idxs
    def apply_collision(self, collision_idx):
        i, j = collision_idx
        collision_dx = self.collision_finder.dx[i, j] + self.dt * self.collision_finder.dv[i, j]
        collision_dir = collision_dx / self.d
        momentum_exchange = np.dot(self.collision_finder.dv[i, j], collision_dir) * collision_dir
        self.v[i] += - momentum_exchange
        self.v[j] += momentum_exchange
    def time_step(self):
        self.prepare_step()
        next_collision_i = self.find_next_collision()
        self.linear_time_step()
        if(next_collision_i is not None):
            self.apply_collision(next_collision_i)
        return self.dt



def make_collision_graph(collisions: list):
    particles = list(set(c[0][0] for c in collisions).union(set(c[0][1] for c in collisions)))
    start_nodes = { i: f'start_{i}' for i in particles }
    end_nodes = { i: f'end_{i}' for i in particles }
    collision_nodes = { i: f'c_{i}' for i in range(len(collisions)) }
    nodes = list(start_nodes.values()) + list(collision_nodes.values()) + list(end_nodes.values())
    edges = []
    edge_i = 0
    particle_lines = {
        n: []
        for n in start_nodes
    }
    def create_in_edge(c_i, p_i, edge_i):
        line_of_p_i = particle_lines[p_i]
        if(line_of_p_i == []):
            edge_start = start_nodes[p_i]
            edges.append((edge_start, collision_nodes[c_i]))
            line_of_p_i.append(edge_i)
            edge_i += 1
        else:
            open_edge = edges[line_of_p_i[-1]]
            closed_edge = (open_edge[0], collision_nodes[c_i])
            edges[line_of_p_i[-1]] = closed_edge
        return edge_i
    #
    def create_out_edge(c_i, p_i, edge_i):
        edge = (collision_nodes[c_i], end_nodes[p_i])
        edges.append(edge)
        particle_lines[p_i].append(edge_i)
        return edge_i+1
    #
    for c_i, collision in enumerate(collisions):
        (p_i, p_j), _ = collision
        edge_i = create_in_edge(c_i, p_i, edge_i)
        edge_i = create_in_edge(c_i, p_j, edge_i)
        edge_i = create_out_edge(c_i, p_i, edge_i)
        edge_i = create_out_edge(c_i, p_j, edge_i)
    return nodes, edges



def generate_dot_file(edges: list[tuple[str, str]], oriented=False):
    # generate a DOT graph
    # digraph collisions {
    #   0 -> 1 [label="123"];
    #   etc.
    # }
    def process_node_name(node_name: str) -> str:
        #return node_name.replace('_', '<sub>') + '</sub>'
        return node_name.replace('_', '')
    if(oriented):
        dot_content = "digraph"
    else:
        dot_content = "graph"
    dot_content += " collisions {\n  rankdir=BT;\n"
    edge_symbol = "->" if(oriented) else "--"
    dot_content += '\n'.join([ f"  {process_node_name(edge[0])} {edge_symbol} {process_node_name(edge[1])};" for edge in edges ])
    dot_content += '\n}\n'
    return dot_content

def compute_marker_size(fig: pyplot.Figure, ax: pyplot.Axes, marker_real_size: float):
    b = ax.transData._b.get_matrix()
    pixels_per_unit_x = b[0,0]
    pixels_per_unit_y = b[1,1]
    marker_size = 17.361 * (fig.dpi**2) * (marker_real_size**2) / (pixels_per_unit_x * pixels_per_unit_y)
    return marker_size


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--T', type=float, default=300., help='temperature (K)')
    arg_parser.add_argument('--m', type=float, default=32., help='molar mass (g/mol)')
    arg_parser.add_argument('--sphere-r', type=float, default=1e-1, help='sphere radius (arbitrary units)')
    arg_parser.add_argument('--n', type=int, default=40, help='number of spheres')
    arg_parser.add_argument('--box-size', type=float, default=20., help='size of the box (arbitrary units) - 0 means no bounding box')
    arg_parser.add_argument('--base-dt', type=float, default=0.001, help='time step size when no collision occur (arbitrary units)')
    arg_parser.add_argument('--dot-file', type=str, default='sample.dot', help='file path to save the collision graph data')
    arg_parser.add_argument('--svg-file', type=str, default='sample.svg', help='file path to save a picture of the collision graph')
    #arg_parser.add_argument('--eps', type=float, default=0., help='unused')
    arg_parser.add_argument('--test', action='store_true', help='run tests')
    parsed_args = arg_parser.parse_args()

    T: float = parsed_args.T
    m: float = parsed_args.m / (1000. * N_avogadro)
    sphere_r: float = parsed_args.sphere_r
    n: int = parsed_args.n
    box_size: float = parsed_args.box_size
    base_dt: float = parsed_args.base_dt
    output_dot_file: str = parsed_args.dot_file
    output_svg_file: str = parsed_args.svg_file
    do_tests: bool = parsed_args.test
    eps: float = 0. # parsed_args.eps

    # tests
    if(do_tests):
        # collision detection
        cf = CollisionFinder(x=np.array([[-1.,0.], [0.,0.]]),
                            v=np.array([[1.,0.],[0.,0.]]),
                            t=2.,
                            d2=0.01)
        assert(np.allclose(cf.dx[0,0],[0.,0.]))
        assert(np.allclose(cf.dx[1,0],[1.,0.]))
        assert(np.allclose(cf.dx[0,1],[-1.,0.]))
        assert(np.allclose(cf.dx[1,1],[0.,0.]))
        assert(np.allclose(cf.dv[0,0],[0.,0.]))
        assert(np.allclose(cf.dv[1,0],[-1.,0.]))
        assert(np.allclose(cf.dv[0,1],[1.,0.]))
        assert(np.allclose(cf.dv[1,1],[0.,0.]))
        assert(np.allclose(cf.dxdv[0,0], 0.))
        assert(np.allclose(cf.dxdv[1,0], -1.))
        assert(np.allclose(cf.dxdv[0,1], -1.))
        assert(np.allclose(cf.dxdv[1,1], 0.))
        cf.get_next_collision()
        # collision data to graph
        collisions = [[(5, 6), 0.0008667870314063772], [(6, 7), 4.531346070838127], [(1, 3), 7.5919666296061115]]
        nodes, edges = make_collision_graph(collisions)
        assert(nodes == ['start_1', 'start_3', 'start_5', 'start_6', 'start_7', 'c_0', 'c_1', 'c_2', 'end_1', 'end_3', 'end_5', 'end_6', 'end_7'])
        assert(edges == [('start_5', 'c_0'), ('start_6', 'c_0'), ('c_0', 'end_5'), ('c_0', 'c_1'), ('start_7', 'c_1'), ('c_1', 'end_6'), ('c_1', 'end_7'), ('start_1', 'c_2'), ('start_3', 'c_2'), ('c_2', 'end_1'), ('c_2', 'end_3')])
        print('all tests passed')




    # simulate hard sphere dynamics & display on a plot as it runs

    v0 = Maxwellian2D(m, T).sample(n)
    x0 = box_size * rng.random(size=[n, 2])
    z0 = np.stack([x0, v0], axis=2)

    dyn = HardSphereDynamics(sphere_r=sphere_r, base_dt=base_dt, z0=z0, box_size=box_size, eps=eps)

    fig, ax = pyplot.subplots(dpi=300)
    ticks = np.arange(start=0, stop=box_size, step=1.)
    ax.set_xlim(left=0., right=box_size)
    ax.set_ylim(bottom=0., top=box_size)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    scatter_plot = ax.scatter(dyn.x[:,0], dyn.x[:,1],
                              edgecolor='k',
                              linewidths=1.)
    if(True):
        marker_size = compute_marker_size(fig, ax, sphere_r)
        ax.clear()
        ax.set_xlim(left=0., right=box_size)
        ax.set_ylim(bottom=0., top=box_size)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        scatter_plot = ax.scatter(dyn.x[:,0], dyn.x[:,1],
                                edgecolor='k',
                                linewidths=1.,
                                s=marker_size)

    def update_scatter(frame,
                       scatter_plot: matplotlib.collections.PathCollection = None,
                       dyn: HardSphereDynamics = None):
        dyn.time_step()
        scatter_plot.set_offsets(list(zip(dyn.x[:,0], dyn.x[:,1])))
        time.sleep(0.02)

    animation = FuncAnimation(fig=fig,
                              func=partial(update_scatter, scatter_plot=scatter_plot, dyn=dyn),
                              frames=None)
    pyplot.show()

    collision_times = np.array([c[-1] for c in dyn.collisions])
    collision_counts = np.arange(len(collision_times)) + 1
    pyplot.plot(collision_times, collision_counts)
    pyplot.title('number of collisions')
    pyplot.xlabel('time')
    pyplot.ylabel('collisions')
    pyplot.show()

    # create a graph of the collisions

    nodes, edges = make_collision_graph(dyn.collisions)

    dot_content = generate_dot_file(edges)

    with open(output_dot_file, 'w') as dot_file:
        dot_file.write(dot_content)

    # dot -Tsvg  sample.dot  > sample.svg
    subprocess.run(['dot', '-Tsvg', output_dot_file, '-o', output_svg_file])
