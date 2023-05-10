import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type == "zeroPadding":
        padding_img = np.zeros(np.array(img.shape) + padding_size * 2)
        padding_img[padding_size:-padding_size,
                    padding_size:-padding_size] = img

        return padding_img
    elif type == "replicatePadding":
        padding_img = np.empty(np.array(img.shape) + padding_size * 2)
        padding_img[padding_size:-padding_size,
                    padding_size:-padding_size] = img

        # up
        padding_img[:padding_size, padding_size:-
                    padding_size] = img[0]
        padding_img[:padding_size,
                    :padding_size] = img[0, 0]

        # left
        padding_img[padding_size:-padding_size,
                    : padding_size] = img[:, 0][:, np.newaxis]
        padding_img[-padding_size:,
                    :padding_size] = img[-1, 0]

        # down
        padding_img[-padding_size:, padding_size:-
                    padding_size] = img[-1]
        padding_img[-padding_size:,
                    -padding_size:] = img[-1, -1]

        # right
        padding_img[padding_size:-padding_size,
                    -padding_size:] = img[:, -1][:, np.newaxis]
        padding_img[:padding_size,
                    -padding_size:] = img[0, -1]

        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding
    padding_img = padding(img, 1, "zeroPadding")

    # build the Toeplitz matrix and compute convolution

    # basic element: 6x8 matrix
    rows = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    cols = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]

    block1 = np.zeros((6, 8))
    block1[rows, cols] = np.tile(kernel[0], (6, 1))

    block2 = np.zeros((6, 8))
    block2[rows, cols] = np.tile(kernel[1], (6, 1))

    block3 = np.zeros((6, 8))
    block3[rows, cols] = np.tile(kernel[2], (6, 1))

    # use Kronecker product to get the diagonal block matrix
    I6 = np.eye(6)
    block1, block2, block3 = np.kron(I6, block1), np.kron(
        I6, block2), np.kron(I6, block3)

    # get the Toeplitz matrix by shifting the diagonal block matrices
    zero = np.zeros((36, 8))
    toeplitz = np.concatenate([block1, zero, zero], 1) + np.concatenate(
        [zero, block2, zero], 1) + np.concatenate([zero, zero, block3], 1)

    # toeplitz_matrix @ flattened padding_imgae
    temp_result = toeplitz @ np.reshape(padding_img, 64)
    output = np.reshape(temp_result, (6, 6))

    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """

    # build the sliding-window convolution here
    nH, nW = img.shape
    kH, kW = kernel.shape
    output_H, output_W = nH-kH+1, nW-kW+1

    # construct the index
    # select the rows
    i1 = np.repeat(np.arange(kH), kW)
    i2 = np.repeat(np.arange(output_H), output_W)
    i = i1.reshape((-1, 1)) + i2.reshape((1, -1))
    # select the columns
    j1 = np.tile(np.arange(kW), kH)
    j2 = np.tile(np.arange(output_W), output_H)
    j = j1.reshape((-1, 1)) + j2.reshape((1, -1))

    # convolution
    conv_matrix = img[i, j]  # shape: (k^2, (N-k+1)^2)
    output = kernel.reshape((1, -1)) @ conv_matrix
    output = output.reshape((output_H, output_W))

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array(
        [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)
