import torch
import warnings
import sys
import os
import argparse

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.tokenizer_utils import build_tokenizer
from model.LMConfig import LMConfig
from model.model import MiniLLMLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers(torch_path, transformers_path, tokenizer_dir):
    def export_tokenizer(transformers_path):
        tokenizer = build_tokenizer(tokenizer_dir, trust_remote_code=True)
        tokenizer.save_pretrained(transformers_path)

    LMConfig.register_for_auto_class()
    MiniLLMLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniLLMLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    export_tokenizer(transformers_path)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


# don't need to use
def push_to_hf(export_model_path):
    def init_model():
        tokenizer = AutoTokenizer.from_pretrained('../model/minillm_tokenizer')
        model = AutoModelForCausalLM.from_pretrained(export_model_path, trust_remote_code=True)
        return model, tokenizer

    model, tokenizer = init_model()
    # model.push_to_hub(model_path)
    # tokenizer.push_to_hub(model_path, safe_serialization=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MiniLLM between torch and transformers formats')
    parser.add_argument('--mode', choices=['torch2hf', 'hf2torch'], default='torch2hf')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=8192)
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--torch_path', type=str, default=None, help='Input torch checkpoint path or output path for hf2torch')
    parser.add_argument('--transformers_path', type=str, default=None, help='Output dir for HF model or input dir for hf2torch')
    parser.add_argument('--tokenizer_dir', type=str, default='../model/minillm_tokenizer', help='Tokenizer source (local dir or HF repo id)')
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    # defaults
    if args.mode == 'torch2hf':
        if args.torch_path is None:
            args.torch_path = f"../out/rlhf_{lm_config.dim}{'_moe' if lm_config.use_moe else ''}.pth"
        if args.transformers_path is None:
            args.transformers_path = '../MiniLLM2-Small'
        convert_torch2transformers(args.torch_path, args.transformers_path, args.tokenizer_dir)
    else:
        if args.transformers_path is None:
            args.transformers_path = '../MiniLLM2-Small'
        if args.torch_path is None:
            args.torch_path = f"../out/rlhf_{lm_config.dim}{'_moe' if lm_config.use_moe else ''}.pth"
        convert_transformers2torch(args.transformers_path, args.torch_path)
