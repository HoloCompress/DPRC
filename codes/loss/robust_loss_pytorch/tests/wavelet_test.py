# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for wavelet.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import numpy as np
import PIL.Image
import scipy.io
import torch
import robust_loss_pytorch
from robust_loss_pytorch import util
from robust_loss_pytorch import wavelet


class TestWavelet(parameterized.TestCase):

  def setUp(self):
    super(TestWavelet, self).setUp()
    np.random.seed(0)

  def _assert_pyramids_close(self, x0, x1, epsilon):
    """A helper function for assering that two wavelet pyramids are close."""
    if isinstance(x0, tuple) or isinstance(x0, list):
      assert isinstance(x1, (list, tuple))
      assert len(x0) == len(x1)
      for y0, y1 in zip(x0, x1):
        self._assert_pyramids_close(y0, y1, epsilon)
    else:
      assert not isinstance(x1, (list, tuple))
      np.testing.assert_equal(x0.shape, x1.shape)
      np.testing.assert_allclose(x0, x1, atol=epsilon, rtol=epsilon)

  def testPadWithOneReflectionIsCorrect(self):
    """Tests that pad_reflecting(p) matches np.pad(p) when p is small."""
    for _ in range(4):
      n = int(np.ceil(np.random.uniform() * 8)) + 1
      x = np.random.uniform(size=(n, n, n))
      padding_below = int(np.round(np.random.uniform() * (n - 1)))
      padding_above = int(np.round(np.random.uniform() * (n - 1)))
      axis = int(np.floor(np.random.uniform() * 3.))

      if axis == 0:
        reference = np.pad(x, [[padding_below, padding_above], [0, 0], [0, 0]],
                           'reflect')
      elif axis == 1:
        reference = np.pad(x, [[0, 0], [padding_below, padding_above], [0, 0]],
                           'reflect')
      elif axis == 2:
        reference = np.pad(x, [[0, 0], [0, 0], [padding_below, padding_above]],
                           'reflect')

      result = wavelet.pad_reflecting(x, padding_below, padding_above, axis)
      np.testing.assert_equal(result.shape, reference.shape)
      np.testing.assert_equal(result, reference)

  def testPadWithManyReflectionsIsCorrect(self):
    """Tests that pad_reflecting(k * p) matches np.pad(p) applied k times."""
    for _ in range(4):
      n = int(np.random.uniform() * 8.) + 1
      p = n - 1
      x = np.random.uniform(size=(n))
      reference1 = np.pad(x, [[p, p]], 'reflect')
      reference2 = np.pad(reference1, [[p, p]], 'reflect')
      reference3 = np.pad(reference2, [[p, p]], 'reflect')
      result1 = wavelet.pad_reflecting(x, p, p, 0)
      result2 = wavelet.pad_reflecting(x, 2 * p, 2 * p, 0)
      result3 = wavelet.pad_reflecting(x, 3 * p, 3 * p, 0)
      np.testing.assert_equal(result1.shape, reference1.shape)
      np.testing.assert_equal(result1, reference1)
      np.testing.assert_equal(result2.shape, reference2.shape)
      np.testing.assert_equal(result2, reference2)
      np.testing.assert_equal(result3.shape, reference3.shape)
      np.testing.assert_equal(result3, reference3)

  def testPadWithManyReflectionsGolden1IsCorrect(self):
    """Tests pad_reflecting() against a golden example."""
    n = 8
    p0 = 17
    p1 = 13
    x = np.arange(n)
    reference = np.concatenate(
        (np.arange(3, 0, -1), np.arange(n), np.arange(n - 2, 0,
                                                      -1), np.arange(n),
         np.arange(n - 2, 0, -1), np.arange(7)))
    result = wavelet.pad_reflecting(x, p0, p1, 0)
    np.testing.assert_equal(result.shape, reference.shape)
    np.testing.assert_equal(result, reference)

  def testPadWithManyReflectionsGolden2IsCorrect(self):
    """Tests pad_reflecting() against a golden example."""
    n = 11
    p0 = 15
    p1 = 7
    x = np.arange(n)
    reference = np.concatenate(
        (np.arange(5, n), np.arange(n - 2, 0,
                                    -1), np.arange(n), np.arange(n - 2, 2, -1)))
    result = wavelet.pad_reflecting(x, p0, p1, 0)
    np.testing.assert_equal(result.shape, reference.shape)
    np.testing.assert_equal(result, reference)

  def testAnalysisLowpassFiltersAreNormalized(self):
    """Tests that the analysis lowpass filter doubles the input's magnitude."""
    for wavelet_type in wavelet.generate_filters():
      filters = wavelet.generate_filters(wavelet_type)
      # The sum of the outer product of the analysis lowpass filter with itself.
      magnitude = np.sum(filters.analysis_lo[:, np.newaxis] *
                         filters.analysis_lo[np.newaxis, :])
      np.testing.assert_allclose(magnitude, 2., atol=1e-10, rtol=1e-10)

  def testWaveletTransformationIsVolumePreserving(self):
    """Tests that construct() is volume preserving when size is a power of 2."""
    for wavelet_type in wavelet.generate_filters():
      sz = (1, 4, 4)
      num_levels = 2
      # Construct the Jacobian of construct().
      im = np.float32(np.random.uniform(0., 1., sz))
      jacobian = []
      vec = lambda x: torch.reshape(x, [-1])
      for d in range(im.size):
        var_im = torch.autograd.Variable(torch.tensor(im), requires_grad=True)
        coeff = vec(
            wavelet.flatten(
                wavelet.construct(var_im, num_levels, wavelet_type)))[d]
        coeff.backward()
        jacobian.append(np.reshape(var_im.grad.detach().numpy(), [-1]))
      jacobian = np.stack(jacobian, 1)
      # Assert that the determinant of the Jacobian is close to 1.
      det = np.linalg.det(jacobian)
      np.testing.assert_allclose(det, 1., atol=1e-5, rtol=1e-5)

  def _load_golden_data(self):
    """Loads golden data: an RGBimage and its CDF9/7 decomposition.

    This golden data was produced by running the code from
    https://www.getreuer.info/projects/wavelet-cdf-97-implementation
    on a test image.

    Returns:
      A tuple containing and image, its decomposition, and its wavelet type.
    """
    golden_filename = os.path.join(
      os.path.dirname(__file__), 'resources', 'wavelet_golden.mat')
    data = scipy.io.loadmat(golden_filename)
    im = np.float32(data['I_color'])
    pyr_true = data['pyr_color'][0, :].tolist()
    for i in range(len(pyr_true) - 1):
      pyr_true[i] = tuple(pyr_true[i].flatten())
    pyr_true = tuple(pyr_true)
    wavelet_type = 'CDF9/7'
    return im, pyr_true, wavelet_type

  @parameterized.named_parameters(('CPU', 'cpu'), ('GPU', 'cuda'))
  def testConstructMatchesGoldenData(self, device):
    """Tests construct() against golden data."""
    im, pyr_true, wavelet_type = self._load_golden_data()
    im = torch.tensor(im, device=device)
    pyr = wavelet.construct(im, len(pyr_true) - 1, wavelet_type)

    pyr = list(pyr)
    for d in range(len(pyr) - 1):
      pyr[d] = list(pyr[d])
      for b in range(3):
        pyr[d][b] = pyr[d][b].cpu().detach()
    d = len(pyr) - 1
    pyr[d] = pyr[d].cpu().detach()

    self._assert_pyramids_close(pyr, pyr_true, 1e-5)

  @parameterized.named_parameters(('CPU', 'cpu'), ('GPU', 'cuda'))
  def testCollapseMatchesGoldenData(self, device):
    """Tests collapse() against golden data."""
    im, pyr_true, wavelet_type = self._load_golden_data()

    pyr_true = list(pyr_true)
    for d in range(len(pyr_true) - 1):
      pyr_true[d] = list(pyr_true[d])
      for b in range(3):
        pyr_true[d][b] = torch.tensor(pyr_true[d][b], device=device)
    d = len(pyr_true) - 1
    pyr_true[d] = torch.tensor(pyr_true[d], device=device)

    recon = wavelet.collapse(pyr_true, wavelet_type).cpu().detach()
    np.testing.assert_allclose(recon, im, atol=1e-5, rtol=1e-5)

  def testVisualizeMatchesGoldenData(self):
    """Tests visualize() (and implicitly flatten())."""
    _, pyr, _ = self._load_golden_data()
    vis = wavelet.visualize(pyr).detach().numpy()
    golden_vis_filename = os.path.join(
      os.path.dirname(__file__), 'resources', 'wavelet_vis_golden.png')
    vis_true = np.asarray(PIL.Image.open(golden_vis_filename))
    # Allow for some slack as quantization may exaggerate some errors.
    np.testing.assert_allclose(
        np.float32(vis_true) / 255.,
        np.float32(vis) / 255.,
        atol=0.005,
        rtol=0.005)

  def testAccurateRoundTripWithSmallRandomImages(self):
    """Tests that collapse(construct(x)) == x for x = [1, k, k], k in [1, 4]."""
    for wavelet_type in wavelet.generate_filters():
      for width in range(0, 5):
        sz = [1, width, width]
        num_levels = wavelet.get_max_num_levels(sz)
        im = np.random.uniform(size=sz)

        pyr = wavelet.construct(im, num_levels, wavelet_type)
        recon = wavelet.collapse(pyr, wavelet_type)
        np.testing.assert_allclose(recon, im, atol=1e-8, rtol=1e-8)

  def testAccurateRoundTripWithLargeRandomImages(self):
    """Tests that collapse(construct(x)) == x for large random x's."""
    for wavelet_type in wavelet.generate_filters():
      for _ in range(4):
        num_levels = np.int32(np.ceil(4 * np.random.uniform()))
        sz_clamp = 2**(num_levels - 1) + 1
        sz = np.maximum(
            np.int32(
                np.ceil(np.array([2, 32, 32]) * np.random.uniform(size=3))),
            np.array([0, sz_clamp, sz_clamp]))
        im = np.random.uniform(size=sz)
        pyr = wavelet.construct(im, num_levels, wavelet_type)
        recon = wavelet.collapse(pyr, wavelet_type)
        np.testing.assert_allclose(recon, im, atol=1e-8, rtol=1e-8)

  def testDecompositionIsNonRedundant(self):
    """Test that wavelet construction is not redundant.

    If the wavelet decompositon is not redundant, then we should be able to
    1) Construct a wavelet decomposition
    2) Alter a single coefficient in the decomposition
    3) Collapse that decomposition into an image and back
    and the two wavelet decompositions should be the same.
    """
    for wavelet_type in wavelet.generate_filters():
      for _ in range(4):
        # Construct an image and a wavelet decomposition of it.
        num_levels = np.int32(np.ceil(4 * np.random.uniform()))
        sz_clamp = 2**(num_levels - 1) + 1
        sz = np.maximum(
            np.int32(
                np.ceil(np.array([2, 32, 32]) * np.random.uniform(size=3))),
            np.array([0, sz_clamp, sz_clamp]))
        im = np.random.uniform(size=sz)
        pyr = wavelet.construct(im, num_levels, wavelet_type)

      # Pick a coefficient at random in the decomposition to alter.
      d = np.int32(np.floor(np.random.uniform() * len(pyr)))
      v = np.random.uniform()
      if d == (len(pyr) - 1):
        if np.prod(pyr[d].shape) > 0:
          c, i, j = np.int32(
              np.floor(np.array(np.random.uniform(size=3)) *
                       pyr[d].shape)).tolist()
          pyr[d][c, i, j] = v
      else:
        b = np.int32(np.floor(np.random.uniform() * len(pyr[d])))
        if np.prod(pyr[d][b].shape) > 0:
          c, i, j = np.int32(
              np.floor(np.array(np.random.uniform(size=3)) *
                       pyr[d][b].shape)).tolist()
          pyr[d][b][c, i, j] = v

      # Collapse and then reconstruct the wavelet decomposition, and check
      # that it is unchanged.
      recon = wavelet.collapse(pyr, wavelet_type)
      pyr_again = wavelet.construct(recon, num_levels, wavelet_type)
      self._assert_pyramids_close(pyr, pyr_again, 1e-8)

  def testUpsampleAndDownsampleAreTransposes(self):
    """Tests that _downsample() is the transpose of _upsample()."""
    n = 8
    x = np.random.uniform(size=(1, n, 1))
    for f_len in range(1, 5):
      f = np.random.uniform(size=f_len)
      for shift in [0, 1]:

        # We're only testing the resampling operators away from the boundaries,
        # as this test appears to fail in the presences of boundary conditions.
        range1 = np.arange(f_len // 2 + 1, n - (f_len // 2 + 1))
        range2 = np.arange(f_len // 4, n // 2 - (f_len // 4))

        y = wavelet._downsample(x, f, 0, shift).detach().numpy()

        vec = lambda x: torch.reshape(x, [-1])

        # Construct the jacobian of _downsample().
        jacobian_down = []
        for d in range2:
          var_x = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
          coeff = vec(wavelet._downsample(var_x, f, 0, shift))[d]
          coeff.backward()
          jacobian_down.append(np.reshape(var_x.grad.detach().numpy(), [-1]))
        jacobian_down = np.stack(jacobian_down, 1)

        # Construct the jacobian of _upsample().
        jacobian_up = []
        for d in range1:
          var_y = torch.autograd.Variable(torch.tensor(y), requires_grad=True)
          coeff = vec(wavelet._upsample(var_y, x.shape[1:], f, 0, shift))[d]
          coeff.backward()
          jacobian_up.append(np.reshape(var_y.grad.detach().numpy(), [-1]))
        jacobian_up = np.stack(jacobian_up, 1)

        # Test that the jacobian of _downsample() is close to the transpose of
        # the jacobian of _upsample().
        np.testing.assert_allclose(
            jacobian_down[range1, :],
            np.transpose(jacobian_up[range2, :]),
            atol=1e-6,
            rtol=1e-6)

  def _construct_preserves_dtype(self, float_dtype):
    """Checks that construct()'s output has the same precision as its input."""
    x = float_dtype(np.random.normal(size=(3, 16, 16)))
    for wavelet_type in wavelet.generate_filters():
      y = wavelet.flatten(wavelet.construct(x, 3, wavelet_type))
      np.testing.assert_equal(y.detach().numpy().dtype, float_dtype)

  def testConstructPreservesDtypeSingle(self):
    self._construct_preserves_dtype(np.float32)

  def testConstructPreservesDtypeDouble(self):
    self._construct_preserves_dtype(np.float64)

  def _collapse_preserves_dtype(self, float_dtype):
    """Checks that collapse()'s output has the same precision as its input."""
    n = 16
    x = []
    for n in [8, 4, 2]:
      band = []
      for _ in range(3):
        band.append(float_dtype(np.random.normal(size=(3, n, n))))
      x.append(band)
    x.append(float_dtype(np.random.normal(size=(3, n, n))))
    for wavelet_type in wavelet.generate_filters():
      y = wavelet.collapse(x, wavelet_type)
      np.testing.assert_equal(y.detach().numpy().dtype, float_dtype)

  def testCollapsePreservesDtypeSingle(self):
    self._collapse_preserves_dtype(np.float32)

  def testCollapsePreservesDtypeDouble(self):
    self._collapse_preserves_dtype(np.float64)

  def testRescaleOneIsANoOp(self):
    """Tests that rescale(x, 1) = x."""
    im = np.random.uniform(size=(2, 32, 32))
    pyr = wavelet.construct(im, 4, 'LeGall5/3')
    pyr_rescaled = wavelet.rescale(pyr, 1.)
    self._assert_pyramids_close(pyr, pyr_rescaled, 1e-8)

  def testRescaleDoesNotAffectTheFirstLevel(self):
    """Tests that rescale(x, s)[0] = x[0] for any s."""
    im = np.random.uniform(size=(2, 32, 32))
    pyr = wavelet.construct(im, 4, 'LeGall5/3')
    pyr_rescaled = wavelet.rescale(pyr, np.exp(np.random.normal()))
    self._assert_pyramids_close(pyr[0:1], pyr_rescaled[0:1], 1e-8)

  def testRescaleOneHalfIsNormalized(self):
    """Tests that rescale(construct(k), 0.5)[-1] = k for constant image k."""
    for num_levels in range(5):
      k = np.random.uniform()
      im = k * np.ones((2, 32, 32))
      pyr = wavelet.construct(im, num_levels, 'LeGall5/3')
      pyr_rescaled = wavelet.rescale(pyr, 0.5)
      np.testing.assert_allclose(
          pyr_rescaled[-1],
          k * np.ones_like(pyr_rescaled[-1]),
          atol=1e-8,
          rtol=1e-8)

  def testRescaleAndUnrescaleReproducesInput(self):
    """Tests that rescale(rescale(x, k), 1/k) = x."""
    im = np.random.uniform(size=(2, 32, 32))
    scale_base = np.exp(np.random.normal())
    pyr = wavelet.construct(im, 4, 'LeGall5/3')
    pyr_rescaled = wavelet.rescale(pyr, scale_base)
    pyr_recon = wavelet.rescale(pyr_rescaled, 1. / scale_base)
    self._assert_pyramids_close(pyr, pyr_recon, 1e-8)


if __name__ == '__main__':
  np.testing.run_module_suite()
