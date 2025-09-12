"""Public exports for the modules subpackage.

Uses relative imports so it works when the parent package is imported
instead of relying on execution from the directory.

Note: Removed the old 'utils' wildcard import because there is no
utils.py in this directory; it caused ImportError in a clean env.
"""

from .activations import (
	SiluFunction,
	GeluFunction,
	DropoutFunction,
	SiLU,
	GELU,
	Dropout,
)
from .linear import (
	TritonMatmulFunction,
	TritonLinear,
)
from .useful_combinations import (
	FusedMLP,
	BaselineMLP,
	TritonMLP,
)
from .attention import (
	MultiHeadAttention,
)
from .basic_block import (
	MiniLLMBlock as TritonMiniLLMBlock,
	MiniLLMLM_Triton,
	MiniLLMConfig,
)

__all__ = [
	# activations
	"SiluFunction", "GeluFunction", "DropoutFunction",
	"SiLU", "GELU", "Dropout",
	# linear
	"TritonMatmulFunction", "TritonLinear",
	# mlp variants
	"FusedMLP", "BaselineMLP", "TritonMLP",
	# attention
	"MultiHeadAttention",
	# minillm style
	"TritonMiniLLMBlock", "MiniLLMLM_Triton", "MiniLLMConfig",
]
