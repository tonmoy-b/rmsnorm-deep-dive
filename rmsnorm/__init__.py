from . import _C

def forward(x, weight, eps=1e-6):
    """Custom RMSNorm forward. Returns (y, rrms)."""
    return _C.forward(x, weight, eps)