import numpy as np
from utils import read_img


if __name__ == "__main__":

    # input
    input_vector = np.zeros((10, 784))
    for i in range(10):
        input_vector[i, :] = read_img(
            "mnist_subset/"+str(i)+".png").reshape(-1)/255.
    gt_y = np.zeros((10, 1))
    gt_y[0] = 1

    np.random.seed(14)

    # Intialization MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784, 16)
    MLP_layer_2 = np.random.randn(16, 1)
    lr = 1e-1
    loss_list = []

    for i in range(50):
        # Forward
        output_layer_1 = input_vector.dot(MLP_layer_1)
        # sigmoid activation function
        output_layer_1_act = 1 / (1+np.exp(-output_layer_1))
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = 1 / (1+np.exp(-output_layer_2))  # sigmoid activation function
        loss = -(gt_y * np.log(pred_y) + (1-gt_y) *
                 np.log(1-pred_y)).sum()  # cross-entroy loss
        print("iteration: %d, loss: %f" % (i+1, loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        # logLoss
        dLdy = -(gt_y / pred_y + (gt_y - 1) / (1-pred_y))
        # sigmoid
        dLdo2 = dLdy * pred_y * (1-pred_y)
        # weights w.r.t output of layer 1
        grad_layer_2 = output_layer_1_act.T @ dLdo2

        # output w.r.t weights
        dLdo1a = dLdo2 @ MLP_layer_2.T
        # sigmoid
        dLdo1 = dLdo1a * output_layer_1_act * (1-output_layer_1_act)
        # weights w.r.t. input
        grad_layer_1 = input_vector.T @ dLdo1a

        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2

    np.savetxt("result/HM1_BP.txt", loss_list)
