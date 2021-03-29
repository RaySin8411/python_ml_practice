import numpy as np
import scipy.signal


def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
    res = []
    for i in range(0, int(len(x) / s), s):
        res.append(np.sum(x_padded[i:i + w_rot.shape[0]] * w_rot))
    return np.array(res)


def conv2d(x, w, p=(0, 0), s=(1, 1)):
    w_rot = np.array(w)[::-1, ::-1]
    x_orig = np.array(x)
    n1 = x_orig.shape[0] + 2 * p[0]
    n2 = x_orig.shape[1] + 2 * p[1]
    x_padded = np.zeros(shape=(n1, n2))
    x_padded[p[0]:p[0] + x_orig.shape[0], p[1]:p[1] + x_orig.shape[1]] = x_orig
    res = []
    for i in range(0, int((x_padded.shape[0] - w_rot.shape[0]) / s[0]) + 1, s[0]):
        res.append([])
        for j in range(0, int((x_padded.shape[1] - w_rot.shape[1]) / s[1]) + 1, s[1]):
            x_sub = x_padded[i:i + w_rot.shape[0], j:j + w_rot.shape[1]]
            res[-1].append(np.sum(x_sub * w_rot))
    return np.array(res)


def conv_1d_example():
    x = [1, 3, 2, 4, 5, 6, 1, 3]
    w = [1, 0, 3, 1, 2]
    print('Conv1d Implementation:', conv1d(x, w, p=2, s=1))
    print('Numpy Results:', np.convolve(x, w, mode='same'))


def conv_2d_example():
    x = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
    w = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
    print('Conv2d Implementation:\n', conv2d(x, w, p=(1, 1), s=(1, 1)))
    print('SciPy Results:\n', scipy.signal.convolve2d(x, w, mode='same'))


if __name__ == '__main__':
    conv_1d_example()
    print("-----------------------------------------------------")
    conv_2d_example()
