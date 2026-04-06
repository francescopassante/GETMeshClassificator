import numpy as np
import potpourri3d as pp3d
import trimesh


class MeshPreprocessor:
    def __init__(self, path):
        self.mesh = self.preprocess_mesh(path)

    def preprocess_mesh(self, mesh_path):
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        # we compute its surface area by summing up the areas of all faces, and then scale it into 1
        area = mesh.area
        if area > 0:
            mesh.apply_scale(1 / area)
        return mesh

    def compute_geodesic_neighborhood(self):

        solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)

        # 3. Compute distances from a source vertex p (index p_idx)
        p_idx = 5000
        distances = solver.compute_distance(p_idx)

        # 4. Select vertices within the 0.2 geodesic radius
        neighbor_indices = np.where(distances <= 0.1)[0]
        return neighbor_indices

    # Function to plot the neighbors of vertex 0:
    def plot_neighbors(self, neighbor_indices):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.mesh.vertices[:, 0],
            self.mesh.vertices[:, 1],
            self.mesh.vertices[:, 2],
            color="lightgray",
            s=10,
        )
        ax.scatter(
            self.mesh.vertices[neighbor_indices, 0],
            self.mesh.vertices[neighbor_indices, 1],
            self.mesh.vertices[neighbor_indices, 2],
            color="red",
            s=10,
        )

        # Plot the source vertex in blue
        ax.scatter(
            self.mesh.vertices[5000, 0],
            self.mesh.vertices[5000, 1],
            self.mesh.vertices[5000, 2],
            color="blue",
            s=50,
        )
        ax.set_title("Geodesic Neighborhood of Vertex 0")
        plt.show()


if __name__ == "__main__":
    # Example usage
    path = "data/SHREC11_test_database_new/T42.off"  # Replace with your mesh file path
    preprocessor = MeshPreprocessor(path)
    neighbours = preprocessor.compute_geodesic_neighborhood()
    # Use function to plot the neighbors of vertex 0:
    preprocessor.plot_neighbors(neighbours)
