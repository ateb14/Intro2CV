import numpy as np
from utils import read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x, Sobel_filter_y, padding


def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.

    I_x, I_y = Sobel_filter_x(input_img), Sobel_filter_y(input_img)
    I_xx, I_yy, I_xy = I_x * I_x, I_y * I_y, I_x * I_y
    W = np.ones((window_size, window_size))

    # convolution
    wI_xx, wI_yy, wI_xy = convolve(padding(I_xx, window_size // 2, "replicatePadding"), W), convolve(padding(
        I_yy, window_size // 2, "replicatePadding"), W), convolve(padding(I_xy, window_size // 2, "replicatePadding"), W)

    # calculate the determinant and the trace of M(x,y) at each point
    det = wI_xx * wI_yy - wI_xy * wI_xy
    trace = wI_xx + wI_yy

    theta = det - alpha * trace * trace
    corner_x, corner_y = np.where(theta > threshold)
    filter_theta = theta[theta > threshold].reshape(-1, 1)

    corner_list = np.concatenate(
        (corner_x.reshape(-1, 1), corner_y.reshape(-1, 1), filter_theta), axis=1)

    # the corners in corne_list: a tuple of (index of rows, index of cols, theta)
    return corner_list


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("hand_writting.png")/255.

    # you can adjust the parameters to fit your own implementation
    window_size = 5
    alpha = 0.05
    threshold = 10

    corner_list = corner_response_function(
        input_img, window_size, alpha, threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key=lambda x: x[2], reverse=True)
    NML_selected = []
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted:
        for j in NML_selected:
            if (abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis):
                break
        else:
            NML_selected.append(i[:-1])

    # save results
    draw_corner("hand_writting.png",
                "result/HM1_HarrisCorner.png", NML_selected)
