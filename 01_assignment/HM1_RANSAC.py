import numpy as np
from utils import draw_save_plane_with_points


if __name__ == "__main__":

    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")

    # RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0

    # more than 99.9% probability at least one hypothesis does not contain any outliers
    min_num = 3
    sample_time = int(
        np.ceil((np.log(0.001) / np.log(1-(100/130) ** min_num))))
    distance_threshold = 0.05
    # sample points group
    index = np.random.randint(0, 130, (sample_time, min_num))

    # estimate the plane with sampled points group
    # use homogenous coordinates
    noise_points = np.concatenate((noise_points, np.ones((130, 1))), 1)
    samples = noise_points[index]
    # Use SVD to find h s.t. min|Ah| and |h| = 1, where A contains the points
    _, _, vh = np.linalg.svd(samples)
    best_fit = vh[:, -1, :]

    # evaluate inliers (with point-to-plance distance < distance_threshold)
    # prepare for parallel computing
    best_fit_ex = np.repeat(best_fit, 130, 0)
    noise_points_ex = np.tile(noise_points, (sample_time, 1))
    # Given that |best_fit| = 1, we don't have to calculate the denominator
    distances = np.sum(best_fit_ex * noise_points_ex, axis=1)
    distances = distances.reshape((sample_time, -1))
    # select the inliners
    inliner_map = np.where(distances < distance_threshold, 1, 0)
    inliner_count = np.sum(inliner_map, axis=1)
    best_idx = np.argmax(inliner_count)
    best_inliner_idx = np.where(inliner_map[best_idx] == 1)
    best_inliners = noise_points[best_inliner_idx]

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    _, _, vh = np.linalg.svd(best_inliners)
    pf = vh[-1, :]

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0

    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
