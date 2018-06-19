import unittest
import torch
import numpy as np
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling, ComplexMultiply

torch.manual_seed(0)

def bilinear_pooling(x,y):
    x_size = x.size()
    y_size = y.size()

    assert(x_size[:-1] == y_size[:-1])

    out_size = list(x_size)
    out_size[-1] = x_size[-1]*y_size[-1]

    x = x.view([-1,x_size[-1]])
    y = y.view([-1,y_size[-1]])

    out_stack = []
    for i in range(x.size()[0]):
        out_stack.append(torch.ger(x[i],y[i]))

    out = torch.stack(out_stack)

    return out.view(out_size)

class TestComplexMultiply(unittest.TestCase):
    def test_gradients(self):
        x = (torch.rand(4,128).requires_grad_(),
             torch.rand(4,128).requires_grad_())
        y = (torch.rand(4,128).requires_grad_(),
             torch.rand(4,128).requires_grad_())
        self.assertTrue(torch.autograd.gradcheck(ComplexMultiply.apply, x+y, eps=1))

class TestCompactBilinearPooling(unittest.TestCase):
    def test_pooling(self):
        mcb = CompactBilinearPooling(2048, 2048, 16000)

        # Create 4 arrays of positive reals
        x = torch.rand(4,2048)
        y = torch.rand(4,2048)
        z = torch.rand(4,2048)
        w = torch.rand(4,2048)
        
        # Compute the real bilinear pooling for each pair of array
        bp_xy = bilinear_pooling(x,y).cpu().numpy()
        bp_zw = bilinear_pooling(z,w).cpu().numpy()
        
        # Compute the dot product of the result
        kernel_bp = np.sum(bp_xy*bp_zw, axis=1)
        

        # Repeat the computation with compact bilinear pooling
        cbp_xy = mcb(x,y).cpu().numpy()
        cbp_zw = mcb(z,w).cpu().numpy()
        
        kernel_cbp = np.sum(cbp_xy*cbp_zw, axis=1)

        # The ratio between the two dot product should be close to one.
        ratio = kernel_cbp / kernel_bp

        np.testing.assert_almost_equal(ratio, np.ones_like(ratio), decimal=1)
        
    def test_gradients(self):
        cbp = CompactBilinearPooling(128, 128, 160)
        x = torch.rand(4,128).requires_grad_()
        y = torch.rand(4,128).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(cbp, (x,y), eps=1))

class TestCompactBilinearDoublePooling(unittest.TestCase):
    def test_pooling(self):
        mcb = CompactBilinearPooling(2048, 2048, 16000).double()

        # Create 4 arrays of positive reals
        x = torch.rand(4,2048).double()
        y = torch.rand(4,2048).double()
        z = torch.rand(4,2048).double()
        w = torch.rand(4,2048).double()

        # Compute the real bilinear pooling for each pair of array
        bp_xy = bilinear_pooling(x,y).cpu().numpy()
        bp_zw = bilinear_pooling(z,w).cpu().numpy()

        # Compute the dot product of the result
        kernel_bp = np.sum(bp_xy*bp_zw, axis=1)


        # Repeat the computation with compact bilinear pooling
        cbp_xy = mcb(x,y).cpu().numpy()
        cbp_zw = mcb(z,w).cpu().numpy()

        kernel_cbp = np.sum(cbp_xy*cbp_zw, axis=1)

        # The ratio between the two dot product should be close to one.
        ratio = kernel_cbp / kernel_bp

        np.testing.assert_almost_equal(ratio, np.ones_like(ratio), decimal=1)

    def test_gradients(self):
        cbp = CompactBilinearPooling(128, 128, 160).double()
        x = torch.rand(4,128).double().requires_grad_()
        y = torch.rand(4,128).double().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(cbp, (x,y), eps=1))

if __name__ == '__main__':
    unittest.main()
