# Compact Bilinear Pooling for PyTorch.

This repository has a pure Python implementation of Compact Bilinear Pooling and Count Sketch for PyTorch.

This version relies on the FFT implementation provided with PyTorch 0.4.0 onward. For older versions of PyTorch, use the tag [v0.3.0](https://github.com/gdlg/pytorch_compact_bilinear_pooling/releases/tag/v0.3.0).

## Installation

Run the `setup.py`, for instance:
```
python setup.py install
```

## Usage

`class compact_bilinear_pooling.CompactBilinearPooling(input1_size, input2_size, output_size, h1 = None, s1 = None, h2 = None, s2 = None)`

Basic usage:
```
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

input_size = 2048
output_size = 16000
mcb = CompactBilinearPooling(input_size, input_size, output_size).cuda()
x = torch.rand(4,input_size).cuda()
y = torch.rand(4,input_size).cuda()

z = mcb(x,y)
```

## Test

A couple of test of the implementation of Compact Bilinear Pooling and its gradient can be run using:
```
python test.py
```

## References

 - Yang Gao et al. "[Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition](https://arxiv.org/abs/1511.06062)", 2016
 - Akira Fukui et al. "[Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding](https://arxiv.org/abs/1606.01847)", 2016

