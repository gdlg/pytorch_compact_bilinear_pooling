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
        x = (torch.autograd.Variable(torch.rand(4,128).cuda(), requires_grad=True),
             torch.autograd.Variable(torch.rand(4,128).cuda(), requires_grad=True))
        y = (torch.autograd.Variable(torch.rand(4,128).cuda(), requires_grad=True),
             torch.autograd.Variable(torch.rand(4,128).cuda(), requires_grad=True))
        self.assertTrue(torch.autograd.gradcheck(ComplexMultiply.apply, x+y, eps=1))

class TestCompactBilinearPooling(unittest.TestCase):
    def test_pooling(self):
        mcb = CompactBilinearPooling(2048, 2048, 16000).cuda()

        # Create 4 arrays of positive reals
        x = torch.autograd.Variable(torch.rand(4,2048).cuda(), requires_grad=True)
        y = torch.autograd.Variable(torch.rand(4,2048).cuda(), requires_grad=True)
        z = torch.autograd.Variable(torch.rand(4,2048).cuda(), requires_grad=True)
        w = torch.autograd.Variable(torch.rand(4,2048).cuda(), requires_grad=True)
        
        # Compute the real bilinear pooling for each pair of array
        bp_xy = bilinear_pooling(x,y).data.cpu().numpy()
        bp_zw = bilinear_pooling(z,w).data.cpu().numpy()
        
        # Compute the dot product of the result
        kernel_bp = np.sum(bp_xy*bp_zw, axis=1)
        

        # Repeat the computation with compact bilinear pooling
        cbp_xy = mcb(x,y).data.cpu().numpy()
        cbp_zw = mcb(z,w).data.cpu().numpy()
        
        kernel_cbp = np.sum(cbp_xy*cbp_zw, axis=1)

        # The ratio between the two dot product should be close to one.
        ratio = kernel_cbp / kernel_bp

        np.testing.assert_almost_equal(ratio, np.ones_like(ratio), decimal=1)
        
    def test_gradients(self):
        cbp = CompactBilinearPooling(128, 128, 160).cuda()
        x = torch.autograd.Variable(torch.rand(4,128).cuda(), requires_grad=True)
        y = torch.autograd.Variable(torch.rand(4,128).cuda(), requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(cbp, (x,y), eps=1))

class TestCompactBilinearDoublePooling(unittest.TestCase):
    def test_pooling(self):
        mcb = CompactBilinearPooling(2048, 2048, 16000).double().cuda()

        # Create 4 arrays of positive reals
        x = torch.autograd.Variable(torch.rand(4,2048).double().cuda(), requires_grad=True)
        y = torch.autograd.Variable(torch.rand(4,2048).double().cuda(), requires_grad=True)
        z = torch.autograd.Variable(torch.rand(4,2048).double().cuda(), requires_grad=True)
        w = torch.autograd.Variable(torch.rand(4,2048).double().cuda(), requires_grad=True)

        # Compute the real bilinear pooling for each pair of array
        bp_xy = bilinear_pooling(x,y).data.cpu().numpy()
        bp_zw = bilinear_pooling(z,w).data.cpu().numpy()

        # Compute the dot product of the result
        kernel_bp = np.sum(bp_xy*bp_zw, axis=1)


        # Repeat the computation with compact bilinear pooling
        cbp_xy = mcb(x,y).data.cpu().numpy()
        cbp_zw = mcb(z,w).data.cpu().numpy()

        kernel_cbp = np.sum(cbp_xy*cbp_zw, axis=1)

        # The ratio between the two dot product should be close to one.
        ratio = kernel_cbp / kernel_bp

        np.testing.assert_almost_equal(ratio, np.ones_like(ratio), decimal=1)

    def test_gradients(self):
        cbp = CompactBilinearPooling(128, 128, 160).double().cuda()
        x = torch.autograd.Variable(torch.rand(4,128).double().cuda(), requires_grad=True)
        y = torch.autograd.Variable(torch.rand(4,128).double().cuda(), requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(cbp, (x,y), eps=1))

if __name__ == '__main__':
    unittest.main()
