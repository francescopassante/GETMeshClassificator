import matplotlib.pyplot as plt
import numpy as np
import potpourri3d as pp3d
import torch
import trimesh
from tqdm import tqdm


class MeshPreprocessor:
    def __init__(self, path, subsample=0.1):
        self.mesh = self.preprocess_mesh(path, subsample)

    def preprocess_mesh(self, mesh_path, subsample):
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Subsample the mesh using quadric decimation to reduce the number of vertices while preserving the overall shape
        simplified_mesh = mesh.simplify_quadric_decimation(percent=1 - subsample)

        #  Normalize surface area to 1
        area = simplified_mesh.area
        if area > 0:
            simplified_mesh.apply_scale(1 / np.sqrt(area))

        return simplified_mesh

    def compute_geodesic_neighborhood(self, p_idx, radius):

        solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)

        # 3. Compute distances from a source vertex p (index p_idx)
        distances = solver.compute_distance(p_idx)

        # 4. Select vertices within the 0.2 geodesic radius
        neighbor_indices = np.where(distances <= radius)[0]
        return neighbor_indices

    def compute_log_and_ptransport(self, radius=0.2):
        """
        Efficiently precomputes logarithmic maps and transport angles for
        all neighborhoods in a single pass using the Vector Heat Method.
        """
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        num_vertices = len(vertices)

        # 1. Initialize the Vector Heat Solvers
        # This pre-factors the Laplacian and Poisson matrices once
        dist_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
        vector_solver = pp3d.MeshVectorHeatSolver(vertices, faces)

        # Storage for the sparse neighborhood graph
        # We store these as lists to convert to a unified tensor/format later
        neighbor_data = []

        print(f"Precomputing log and parallel transport for {num_vertices} vertices...")
        for i in tqdm(range(num_vertices)):
            # Identify neighbors within geodesic radius
            dists = dist_solver.compute_distance(i)
            neighbor_indices = np.where(dists <= radius)[0]
            # Remove the center vertex from its own neighborhood
            neighbor_indices = neighbor_indices[neighbor_indices != i]

            # Compute the Logarithmic Map u_q
            # This returns (N, 2) tangent coordinates for ALL vertices.
            # We only keep the ones for the identified neighbors.
            u_q = vector_solver.compute_log_map(i)[neighbor_indices]

            # Compute Parallel Transport g_{q -> p} rotation angles
            # In GET, these align features to the center point's gauge.
            # Note: We compute the angle relative to the local basis.
            g_qp_angles = []
            for q_idx in neighbor_indices:
                # To calculate the parallel transport angle from q to p, we transport a canonical tangent vector (1,0) from q to p and measure its angle in p's frame.
                # transport_tangent_vector(source_index, vector) computes the vector field of transported vector from the source vertex to all others. We only need the one for i.
                transported_v = vector_solver.transport_tangent_vector(
                    q_idx, [1.0, 0.0]
                )[i]

                # The rotation angle is the angle of the transported vector in p's frame
                angle = np.arctan2(transported_v[1], transported_v[0])
                g_qp_angles.append(angle)

            neighbor_data.append(
                {
                    "p_idx": i,
                    "q_indices": neighbor_indices.astype(np.int32),
                    "u_q": u_q.astype(np.float32),
                    "g_qp": np.array(g_qp_angles, dtype=np.float32),
                }
            )

        return neighbor_data

    # Function to plot the neighbors of vertex 0, debug purposes:
    def plot_neighbors(self, p_idx, distance):
        neighbor_indices = self.compute_geodesic_neighborhood(p_idx, distance)
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
            self.mesh.vertices[p_idx, 0],
            self.mesh.vertices[p_idx, 1],
            self.mesh.vertices[p_idx, 2],
            color="blue",
            s=50,
        )
        ax.set_title(f"Geodesic Neighborhood of Vertex {p_idx}")
        plt.show()


if __name__ == "__main__":
    # Example usage
    path = "data/SHREC11_test_database_new/T42.off"  # Replace with your mesh file path
    preprocessor = MeshPreprocessor(path, subsample=0.1)
    print("total vertices: ", len(preprocessor.mesh.vertices))
    neighbor_indices = preprocessor.compute_geodesic_neighborhood(p_idx=100, radius=0.2)
    print("neighbors: ", len(neighbor_indices))
    res = preprocessor.compute_log_and_ptransport(radius=0.2)
    print(res)
