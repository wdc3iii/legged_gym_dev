from deep_tube_learning.datasets import get_dataset
import numpy as np

# data_folder = "hopper_single_int_axxt99kd"
data_folder = "hopper_single_int_yv7pg1wv"

def main():
    dataset = get_dataset(data_folder)

    e = np.linalg.norm(dataset['z'] - dataset['pz_x'], axis=-1).ravel()
    e = np.sort(e)

    print(
        'Quantiles: ',
        '\n10\%: ', e[int(0.1 * e.shape[0])],
        '\n25\%: ', e[int(0.25 * e.shape[0])],
        '\n50\%: ', e[int(0.5 * e.shape[0])],
        '\n75\%: ', e[int(0.75 * e.shape[0])],
        '\n90\%: ', e[int(0.9 * e.shape[0])],
        '\n95\%: ', e[int(0.95 * e.shape[0])],
        '\n99\%: ', e[int(0.99 * e.shape[0])],
        '\n100\%: ', e[-1]
    )

    print('Here')


if __name__ == '__main__':
    main()
