from pydantic import BaseModel, Field
from typing import Optional


class TrainingConfig(BaseModel):
    # Data
    data_path: str = Field(..., description="Path to training JSONL with conversations field")
    test_file: Optional[str] = Field(None, description="Optional test JSONL; if None, split train")
    out_dir: str = Field("out", description="Output directory for checkpoints")

    # Tokenizer & model
    tokenizer_dir: str = Field("./model/minillm_tokenizer", description="Tokenizer directory")
    ckp: Optional[str] = Field(None, description="Pretrained checkpoint to load; if None, random init")

    # LM architecture
    lm_dim: int = Field(512)
    n_layers: int = Field(8)
    n_block: Optional[int] = Field(None)
    max_seq_len: int = Field(512)
    use_moe: bool = Field(False)
    repeat_layer: bool = Field(False)

    # Train hyperparams
    epochs: int = Field(1)
    batch_size: int = Field(32)
    learning_rate: float = Field(5e-5)
    accumulation_steps: int = Field(1)
    grad_clip: float = Field(1.0)
    log_interval: int = Field(100)
    save_interval: int = Field(100)
    max_steps: Optional[int] = Field(None)
    device: str = Field("cuda" , description="Device string e.g. cuda or cpu")
    dtype: str = Field("bfloat16", description="float16|bfloat16|float32")
