import numpy as np
from src.arguments import args
import fenics as fe
import mshr
import matplotlib.pyplot as plt
from matplotlib import collections  as mc


def generate_mesh():
    rec1_points = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1]])
    rec1_points = [fe.Point(p[0], p[1]) for p in rec1_points]
    rec1_dommain = mshr.Polygon(rec1_points)
    
    # theta = np.pi/6.
    theta = np.pi * 0.77

    v1 = [0.5*np.cos(theta + np.pi/2), 0.5*np.sin(theta + np.pi/2)]
    v2 = [2.*np.cos(theta), 2.*np.sin(theta)]
    rec2_points = np.array([[2. - v1[0], 0.5 - v1[1]], 
                            [2. - v1[0] + v2[0], 0.5 - v1[1] + v2[1]],
                            [2. + v1[0] + v2[0], 0.5 + v1[1] + v2[1]],
                            [2. + v1[0], 0.5 + v1[1]]])

    rec2_points = [fe.Point(p[0], p[1]) for p in rec2_points]
    rec2_dommain = mshr.Polygon(rec2_points)

    tri_points = np.array([[2., 0.5], [2., 0.,], [2. - v1[0], 0.5 - v1[1]]])
    tri_points = [fe.Point(p[0], p[1]) for p in tri_points]
    tri_domain = mshr.Polygon(tri_points)

    circle = mshr.Circle(fe.Point(2., 0.5), 0.5)

    domain = rec1_dommain + rec2_dommain + tri_domain

    mesh = mshr.generate_mesh(domain, 10)
    mesh_file =  fe.File(f'data/vtk/u.pvd')
    mesh_file << mesh


def plot_gcode():

    file = open(f'data/gcode/exp.gcode', 'r')
    lines = file.readlines()
    gcode = len(lines)

    print(gcode)
    points = []
    colors = []
 
    for line in lines:
        l = line.split()
        if len(l) == 2:
            if l[1][0] != 'F':
                raise ValueError(f"len = {len(l)}, get {l}")
        elif len(l) == 3:
            if l[2][0] != 'F':
                raise ValueError(f"len = {len(l)}, get {l}")
        elif len(l) == 4:
            if l[3][0] == 'F':
                points.append([float(l[1][1:]), float(l[2][1:])])
                colors.append([1., 0., 0., 1.])
            elif l[3][0] == 'E':
                points.append([float(l[1][1:]), float(l[2][1:])])
                colors.append([0., 0., 1., 1.])                
        else:
            raise ValueError(f"len = {len(l)}, get {l}")

    print(points[:10])
    print(colors[:10])
    points = np.array(points)
    colors = np.array(colors)

    lines = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    colors = colors[1:]

    # lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

    lc = mc.LineCollection(lines, colors=colors, linewidths=2)
    fig, ax = plt.subplots()
    # plt.plot(points[:, 0], points[:, 1], linestyle='None', marker='s', color='black', markersize=5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.axis('equal')




if __name__ == "__main__":
    plot_gcode()
    plt.show()
