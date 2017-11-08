import unittest
import torch
from torch import nn
import numpy as np
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling, ComplexMultiply

torch.manual_seed(0)

class TestMultiGPU(unittest.TestCase):
    def test_multigpu(self):
        mcb = CompactBilinearPooling(2048, 2048, 16000).cuda()
        parallel_mcb = nn.DataParallel(mcb)

        x = torch.autograd.Variable(torch.rand(8,2048).cuda(), requires_grad=True)

        z = parallel_mcb(x)

        z.sum().backward()


if __name__ == '__main__':
    unittest.main()
