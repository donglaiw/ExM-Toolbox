import time
from dataclasses import fields
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import corrpts, utils, optimization, pointcloud


class SimpleICP:
    """Class for setting up and run simpleICP."""

    def __init__(self) -> None:
        """Constructor method."""

        self.pc1 = None
        self.pc2 = None

    def add_point_clouds(
        self,
        pc_fix: pointcloud.PointCloud,
        pc_mov: pointcloud.PointCloud,
    ) -> None:
        """Add fixed and movable point cloud.

        Args:
            pc_fix (pointcloud.PointCloud): Fixed point cloud.
            pc_mov (pointcloud.PointCloud): Movable point cloud. This point cloud
                will be shifted and rotated (transformed) by applying a rigid-body transformation.
        """

        self.pc1 = pc_fix
        self.pc2 = pc_mov

    def run(
        self,
        correspondences: int = 1000,
        neighbors: int = 10,
        min_planarity: float = 0.3,
        max_overlap_distance: float = np.inf,
        min_change: float = 1.0,
        max_iterations: int = 100,
        distance_weights: Optional[float] = 1,
        rbp_observed_values: Tuple[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        rbp_observation_weights: Tuple[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        debug_dirpath: str = "",
    ) -> Tuple[np.array, np.array, optimization.RigidBodyParameters]:
        """Run simpleICP algorithm.

        Args:
            correspondences (int, optional): Number of correspondences selected initially in the
                fixed point cloud for the subsequent point cloud matching. Defaults to 1000.
            neighbors (int, optional): Number of neighboring points used to estimate the normal
                vector and the planarity of the selected points. Defaults to 10.
            min_planarity (float, optional): Minimum planarity value of a correspondence. Defaults
                to 0.3.
            max_overlap_distance (float, optional): Maximum overlap distance between the two point
                clouds. Defaults to np.inf.
            min_change (float, optional): Value to test a convergency criteria after each iteration:
                if the mean and the standard deviation of the point-to-plane distances do not
                change more than min_change (in %), the iteration is stopped. Defaults to 1.0.
            max_iterations (int, optional): Maximum number of ICP iterations. Defaults to 100.
            distance_weights (Optional[float], optional): Weight factor by which the point-to-plane
                residuals are multiplied. Set to None for automatic estimation - however, this makes
                only sense if the rigid-body transformation parameters are observed, see the
                following arguments. Defaults to 1.
            rbp_observed_values (Tuple[float], optional): Values of direct observation of the rigid
                body transformation parameters. These values define the initial transform between
                the two point clouds. Order is alpha1, alpha2, alpha3, tx, ty, tz. Unit of
                alpha1/2/3: degree. Defaults to (0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
            rbp_observation_weights (Tuple[float], optional): Weight factors with which the
                residuals of the direct observations are multiplied. The residuals are defined as
                difference between the estimated rbp and the observed rbp. If an observation weight
                is set to np.inf, this parameter is fixed to the observed value. Order of elements
                is the same as in rbp_observed_values. Defaults to (0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
            debug_dirpath (str, optional): Path to directory for saving debug files, e.g. point
                clouds and correspondences of each iteration. The xyz files can conveniently be
                inspected with CloudCompare. Files containing the string "preoptim" correspond to
                the state before the optimization, i.e. before the movable point cloud is
                transformed. Accordingly, files containing the string "postoptim" correspond to the
                state after the optimization. Not considered if an empty string is passed. Defaults
                to "".
        Returns:
            Tuple[np.array, np.array]:
                H: Estimated homogeneous transformation matrix.
                X_mov_transformed: Points of movable point cloud transformed by H.
                rbp: Data class containing estimates of the rigid-body transformation parameters.
        """

        self.__check_arguments(
            distance_weights, rbp_observed_values, rbp_observation_weights
        )

        start_time = time.time()

        if debug_dirpath:
            print(f'Write debug files to directory "{debug_dirpath}"')
            Path(debug_dirpath).mkdir(parents=True, exist_ok=True)

        # Convert angle valus from degree -> radian
        rbp_observed_values = np.array(rbp_observed_values)
        for i in range(3):
            rbp_observed_values[i] = rbp_observed_values[i] * np.pi / 180

        # Compose initial H from observed rbp values
        H = utils.homo_t_matrix(
            utils.euler_to_rot(
                rbp_observed_values[0], rbp_observed_values[1], rbp_observed_values[2]
            ),
            rbp_observed_values[3:],
        )

        if np.isfinite(max_overlap_distance):

            print("Consider partial overlap of point clouds ...")
            self.pc2.transform_by_H(H)  # temporarily transform pc2
            self.pc1.select_in_range(self.pc2.X, max_range=max_overlap_distance)
            self.pc2.transform_by_H(np.linalg.inv(H))  # undo transformation

            if not self.pc1.num_selected_points > 0:
                raise SimpleICPException(
                    "Point clouds do not overlap within max_overlap_distance = "
                    f"{max_overlap_distance:.5f}! Consider increasing the value of "
                    "max_overlap_distance."
                )

        print("Select points for correspondences in fixed point cloud ...")
        self.pc1.select_n_points(correspondences)
        selected_orig = self.pc1["selected"]

        if not {"nx", "ny", "nz", "planarity"}.issubset(self.pc1.columns):
            print("Estimate normals of selected points ...")
            self.pc1.estimate_normals(neighbors)

        distance_residuals = []

        print("Start iterations ...")
        for it in range(0, max_iterations):

            cp = corrpts.CorrPts(self.pc1, self.pc2)

            self.pc2.transform_by_H(H)  # temporarily transform pc2
            if debug_dirpath:
                if it == 0: # write only in first iteration
                    self.pc1.write_xyz(
                        Path(debug_dirpath).joinpath(
                            f"iteration{it:03d}_preoptim_pcfix.xyz"
                        )
                    )
                self.pc2.write_xyz(
                    Path(debug_dirpath).joinpath(
                        f"iteration{it:03d}_preoptim_pcmov.xyz"
                    )
                )
            cp.match()
            self.pc2.transform_by_H(np.linalg.inv(H))  # undo transformation

            # Rejection of possibly false correspondences
            cp.reject_wrt_planarity(min_planarity)
            cp.reject_wrt_point_to_plane_distances()
            if cp.num_corr_pts < 6:
                raise SimpleICPException(
                    "Too few correspondences! At least 6 correspondences are needed to estimate "
                    "the 6 rigid body transformation parameters. The current number of "
                    f"correspondences is {cp.num_corr_pts}."
                )

            if debug_dirpath:
                cp.write_xyz(
                    Path(debug_dirpath).joinpath(
                        f"iteration{it:03d}_preoptim_correspondences.xyz"
                    )
                )

            if it == 0:
                initial_distances = cp.point_to_plane_distances
                rbp_initial_values = rbp_observed_values
            else:
                rbp_initial_values = rbp_estimated_values

            # Estimate weight of distances if value is None
            if distance_weights is None:
                distance_weights = 1 / (np.std(cp.point_to_plane_distances) ** 2)

            optim = optimization.SimpleICPOptimization(
                cp,
                distance_weights,
                rbp_initial_values,
                rbp_observed_values,
                rbp_observation_weights,
            )

            distance_residuals_new = optim.estimate_parameters()
            rbp = optim.rbp

            rbp_estimated_values = rbp.get_parameter_attributes_as_list(
                "estimated_value"
            )
            H = rbp.H

            distance_residuals.append(distance_residuals_new)

            self.pc1["selected"] = selected_orig  # restore selected points

            if it > 0:
                if self.__check_convergence_criteria(
                    distance_residuals[it], distance_residuals[it - 1], min_change
                ):
                    optim.estimate_parameter_uncertainties()
                    print("Convergence criteria fulfilled -> stop iteration!")
                    break

            if it == 0:
                print(
                    f"{'iteration':>9s} | "
                    f"{'correspondences':>15s} | "
                    f"{'mean(residuals)':>15s} | "
                    f"{'std(residuals)':>15s}"
                )
                print(
                    f"{'orig:0':>9s} | "
                    f"{len(initial_distances):15d} | "
                    f"{np.mean(initial_distances):15.4f} | "
                    f"{np.std(initial_distances):15.4f}"
                )
            print(
                f"{it+1:9d} | "
                f"{len(distance_residuals[it]):15d} | "
                f"{np.mean(distance_residuals[it]):15.4f} | "
                f"{np.std(distance_residuals[it]):15.4f}"
            )

        print("Estimated transformation matrix H:")
        print(f"[{H[0, 0]:12.6f} {H[0, 1]:12.6f} {H[0, 2]:12.6f} {H[0, 3]:12.6f}]")
        print(f"[{H[1, 0]:12.6f} {H[1, 1]:12.6f} {H[1, 2]:12.6f} {H[1, 3]:12.6f}]")
        print(f"[{H[2, 0]:12.6f} {H[2, 1]:12.6f} {H[2, 2]:12.6f} {H[2, 3]:12.6f}]")
        print(f"[{H[3, 0]:12.6f} {H[3, 1]:12.6f} {H[3, 2]:12.6f} {H[3, 3]:12.6f}]")

        print(
            "... which corresponds to the following rigid-body transformation parameters:"
        )
        print(
            f"{'parameter':>9s} | "
            f"{'est.value':>15s} | "
            f"{'est.uncertainty':>15s} | "
            f"{'obs.value':>15s} | "
            f"{'obs.weight':>15s}"
        )
        for parameter in fields(rbp):
            print(
                f"{parameter.name:>9s} | "
                f"{getattr(rbp, parameter.name).estimated_value_scaled:15.6f} | "
                f"{getattr(rbp, parameter.name).estimated_uncertainty_scaled:15.6f} | "
                f"{getattr(rbp, parameter.name).observed_value_scaled:15.6f} | "
                f"{getattr(rbp, parameter.name).observation_weight:15.3e}"
            )

        print(
            "(Unit of est.value, est.uncertainty, and obs.value for alpha1/2/3 is degree)"
        )

        # Apply final transformation
        self.pc2.transform_by_H(H)
        if debug_dirpath:
            self.pc2.write_xyz(
                Path(debug_dirpath).joinpath(f"iteration{it:03d}_postoptim_pcmov.xyz")
            )

        print(f"Finished in {time.time() - start_time:.3f} seconds!")

        return H, self.pc2.X, rbp

    @staticmethod
    def __check_arguments(
        distance_weights, rbp_observed_values, rbp_observation_weights
    ):
        """Some (i.e. not exhaustive) checks of the arguments passed to the constructor method."""
        if distance_weights is not None:
            if distance_weights <= 0:
                raise SimpleICPException("distance_weights must be > 0.")

        if not len(rbp_observed_values) == 6:
            raise SimpleICPException(
                "rbp_observed_values must have exactly 6 elements."
            )

        if not len(rbp_observation_weights) == 6:
            raise SimpleICPException(
                "rbp_observation_weights must have exactly 6 elements."
            )

        if not all([w >= 0 for w in rbp_observation_weights]):
            raise SimpleICPException(
                "All elements of rbp_observation_weights must be >= 0."
            )

        if not any(np.isfinite(rbp_observation_weights)):
            raise SimpleICPException(
                "At least one element in rbp_observation_weights must be finite."
            )

    @staticmethod
    def __check_convergence_criteria(
        distance_residuals_new: np.array,
        distance_residuals_old: np.array,
        min_change: float,
    ) -> bool:
        """Check if the convergence criteria is met."""

        def change(new, old):
            return np.abs((new - old) / old * 100)

        change_of_mean = change(
            np.mean(distance_residuals_new), np.mean(distance_residuals_old)
        )
        change_of_std = change(
            np.std(distance_residuals_new), np.std(distance_residuals_old)
        )

        return (
            True
            if change_of_mean < min_change and change_of_std < min_change
            else False
        )


class SimpleICPException(Exception):
    """The SimpleICP class raises this when the class is misused."""

