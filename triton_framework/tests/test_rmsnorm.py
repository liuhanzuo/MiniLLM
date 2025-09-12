import os
import sys
import shutil
import sysconfig
import torch

# Ensure we can import the package no matter the CWD
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC = os.path.join(ROOT, 'triton_framework', 'src')
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

from triton_framework.modules.simple_lm import RMSNorm, rmsnorm_ref


def _can_triton_jit() -> bool:
    # Require: triton importable, CUDA available, gcc present, Python.h present
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    if shutil.which('gcc') is None:
        return False
    inc = sysconfig.get_paths().get('include')
    if not inc or not os.path.exists(os.path.join(inc, 'Python.h')):
        return False
    return True


def test_rmsnorm_close():
    use_triton = _can_triton_jit()
    print("use triton:", use_triton)
    device = torch.device('cuda') if use_triton else torch.device('cpu')
    x = torch.randn(4, 8, 16, device=device)
    w = torch.ones(16, device=device)

    m = RMSNorm(16, use_triton=use_triton).to(device)
    m.weight.data.copy_(w)

    y = m(x)
    y_ref = rmsnorm_ref(x, w)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4)
if __name__ == '__main__':
    test_rmsnorm_close()
