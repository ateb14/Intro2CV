import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y, padding
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad *
                             y_grad)
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """

    # compute the x and y components of the gradients
    grad_x = np.sin(grad_dir) * grad_mag
    grad_y = np.cos(grad_dir) * grad_mag
    # Surprisingly, mistakenly switching cos and sin leads to a better performance

    # compute the position of the two neighbors of a grid
    x, y = grad_mag.shape
    neighbor_plus_x = np.arange(x).reshape((-1, 1)) + grad_x
    neighbor_plus_y = np.arange(y).reshape((1, -1)) + grad_y
    neighbor_minus_x = np.arange(x).reshape((-1, 1)) - grad_x
    neighbor_minus_y = np.arange(y).reshape((1, -1)) - grad_y

    # bilinear interpolation

    # pad the grad_mag in case that the neighbor is out
    grad_pad = padding(grad_mag, 1, "replicatePadding")

    def bilinear_interpolation(X, Y):
        # compute the indices for the 4 neigbors of the current "neighbor-plus"
        X = X + 1
        X1 = np.floor(X)
        X1 = np.where(X1 >= 1, X1, 0)
        X1 = np.where(X1 <= x, X1, x).astype('int64')
        X2 = (X1 + 1).astype('int64')

        Y = Y + 1
        Y1 = np.floor(Y)
        Y1 = np.where(Y1 >= 1, Y1, 0)
        Y1 = np.where(Y1 <= y, Y1, y).astype('int64')
        Y2 = (Y1 + 1).astype('int64')

        lambda1, lambda2 = X2 - X, X-X1
        nb_plus_interpolation_y1 = lambda1 * \
            grad_pad[X1, Y1] + lambda2*grad_pad[X2, Y1]
        nb_plus_interpolation_y2 = lambda1 * \
            grad_pad[X1, Y2] + lambda2*grad_pad[X2, Y2]

        return (Y2-Y)*nb_plus_interpolation_y1 + (Y-Y1) * nb_plus_interpolation_y2

    # Suppression
    NMS_output = np.where(np.logical_and(bilinear_interpolation(neighbor_plus_x, neighbor_plus_y)
                          <= grad_mag, bilinear_interpolation(neighbor_minus_x, neighbor_minus_y) <= grad_mag), grad_mag, 0)

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    # you can adjust the parameters to fit your own implementation
    low_ratio = 0.64
    high_ratio = 0.85

    x, y = img.shape

    avg_mag = np.sum(img) / np.sum(np.where(img > 0, 1, 0))
    maxVal = high_ratio * avg_mag
    minVal = low_ratio * avg_mag

    # drop-out
    img = np.where(img > minVal, img, 0)

    # reserved
    resv = np.where(img > maxVal, 1, 0)
    output = resv

    dir = [[-1, 0], [-1, 1], [0, 1], [1, 1],
           [1, 0], [1, -1], [0, -1], [-1, -1]]

    change = True
    while change:
        change = False
        for i in range(x):
            for j in range(y):
                if output[i, j]:
                    for pos in range(len(dir)):
                        ii, jj = i + dir[pos][0], j + dir[pos][1]
                        if ii < 0 or ii >= x or jj < 0 or jj >= y:
                            continue
                        if output[ii, jj]:
                            continue
                        if img[ii, jj] > minVal:
                            output[ii, jj] = 1
                            change = True

    return output


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("lenna.png")/255

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    # Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(
        x_grad, y_grad)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    write_img("result/HM1_Canny_result.png", output_img*255)
