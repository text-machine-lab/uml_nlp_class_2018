import numpy as np
import torch
import torch.cuda


def main():
    is_cuda = torch.cuda.is_available()
    print('Is CUDA available:', is_cuda)

    a = torch.from_numpy(np.array([10, 20, 30]))
    b = torch.from_numpy(np.array([1, 2, 3]))

    if is_cuda:
        a = a.cuda()
        b = b.cuda()

    c = a + b

    print('a:', a)
    print('b:', b)
    print('c = a + b:', c)


if __name__ == '__main__':
    main()
