"""Tests for diffusion/distributions.py."""
import math
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.distributions import gen_skewed_levy, gen_sas, match_last_dims


def test_match_last_dims_shape():
    data = torch.arange(4, dtype=torch.float32)
    out = match_last_dims(data, (4, 3, 2))
    assert out.shape == (4, 3, 2)
    assert (out[:, 0, 0] == data).all()


def test_gen_skewed_levy_alpha2():
    a = gen_skewed_levy(2.0, (10,))
    assert a.shape == (10,)
    assert torch.all(a == 2.0)


def test_gen_skewed_levy_shape_isotropic():
    size = (8, 2, 4, 4)
    a = gen_skewed_levy(1.8, size, isotropic=True)
    assert a.shape == torch.Size(size)
    # All elements in same batch should be equal (isotropic)
    for b in range(8):
        assert (a[b] == a[b, 0, 0, 0]).all()


def test_gen_skewed_levy_positive():
    a = gen_skewed_levy(1.5, (100,), isotropic=False)
    assert torch.all(a > 0)


def test_gen_skewed_levy_clamp():
    a = gen_skewed_levy(1.8, (1000,), isotropic=False, clamp_a=5.0)
    assert a.max().item() <= 5.0


def test_gen_sas_shape():
    eps = gen_sas(1.8, (5, 3, 8, 8))
    assert eps.shape == (5, 3, 8, 8)


def test_gen_sas_alpha2_gaussian():
    # alpha=2 → a=2 everywhere, eps = sqrt(2)*G ~ N(0,2) roughly
    eps = gen_sas(2.0, (10000,))
    # Should have finite variance
    assert eps.var().item() < 100.0


def test_gen_sas_with_presampled_a():
    # a must broadcast to size; use same shape as output
    a = torch.ones(4, 2)
    eps = gen_sas(1.8, (4, 2), a=a)
    assert eps.shape == (4, 2)


def test_gen_sas_clamp():
    eps = gen_sas(1.5, (1000,), isotropic=False, clamp_eps=3.0)
    assert eps.abs().max().item() <= 3.0
