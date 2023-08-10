from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
class CGR:
    def __init__(self, n=None, vertex_coordinates=None):
        self.n = n  # number of vertices
        self.vertex_coordinates = vertex_coordinates  # map of <token: coordinates tuple (x,y)>
        self.points = []  # encoded points
        self.current_point = (0, 0)  # initial start point

    def encode(self, sequence, output_filepath):
        for token in sequence:
            # get the coordinates of the vertex corresponding to the current token
            token_vertex_coordinates = self.vertex_coordinates[token]
            x = self.current_point[0] + ((token_vertex_coordinates[0] - self.current_point[0]) / 2)
            y = self.current_point[1] + ((token_vertex_coordinates[1] - self.current_point[1]) / 2)
            self.points.append((x, y))
            self.current_point = (x, y)

        if output_filepath:
            self.save_image(output_filepath)

        return self.points

    def reset(self):
        self.points = []  # encoded points
        self.current_point = (0, 0)  # initial start point

    def save_image(self, output_filepath):
        print(output_filepath)

        plt.clf()
        plt.scatter(*np.array(self.points).T, s=2, c="black")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.savefig(output_filepath)
        # point_coords = self.points + list(self.vertex_coordinates.values())
        # image = Image.fromarray(np.array(point_coords), mode="L")
        # image.save(output_filepath, "PNG")

