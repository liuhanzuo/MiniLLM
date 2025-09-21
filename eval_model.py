import argparse
import random
import time
import os
import numpy as np
import torch
import torch.distributed as dist
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniLLMLM
from model.LMConfig import LMConfig
from model.model_lora import *

warnings.filterwarnings('ignore')


def init_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained('./model/minillm_tokenizer')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        if args.ckp is not None:
            ckp = args.ckp
        else:
            ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        model = MiniLLMLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            n_block=args.n_block,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        # Prefer safe weights-only when available
        try:
            state_dict = torch.load(ckp, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckp, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            './MiniLLM2',
            trust_remote_code=True
        )
    print(f'MiniLLM模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniLLM")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--n_block', default=None, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 携带历史对话上下文条数
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--ckp', default=None, type=str, help="加载指定权重文件，默认None则按模式自动匹配")
    parser.add_argument('--tokenizer_path', default=None, type=str, help="加载指定分词器，默认None则加载minillm_tokenizer")
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
    # Multi-GPU
    parser.add_argument('--ddp', action='store_true', help='使用 torchrun (DDP) 多卡推理，仅 rank0 打印输出')
    parser.add_argument('--local_rank', type=int, default=-1, help='torchrun 会注入 LOCAL_RANK')
    args = parser.parse_args()

    # Setup device/DDP
    world_size = 1
    rank = 0
    use_ddp = False
    if args.ddp:
        assert torch.cuda.is_available(), 'DDP 需要可用的 CUDA 设备'
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank if args.local_rank >= 0 else 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
        use_ddp = True
    else:
        device = torch.device('cuda' if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu')

    model, tokenizer = init_model(args, device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)

    prompts = get_prompt_datas(args)
    # 仅 rank0 读取交互输入，再广播
    if (not use_ddp) or rank == 0:
        test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    else:
        test_mode = 0  # 占位，立即被广播覆盖
    if use_ddp:
        test_mode_tensor = torch.tensor([test_mode], device=device)
        dist.broadcast(test_mode_tensor, src=0)
        test_mode = int(test_mode_tensor.item())
    messages = []
    if test_mode == 0:
        # 自动测试：DDP 下按 rank 切片 prompts，并行生成；仅 rank0 汇总打印
        total = len(prompts)
        indices = list(range(total))
        if use_ddp and world_size > 1:
            # contiguous chunk split
            per = (total + world_size - 1) // world_size
            start = rank * per
            end = min(total, (rank + 1) * per)
            local_pairs = list(zip(indices[start:end], prompts[start:end]))
        else:
            local_pairs = list(zip(indices, prompts))

        local_results = []
        for (idx, prompt) in local_pairs:
            setup_seed(random.randint(0, 2048))
            # 构造消息
            local_messages = messages[-args.history_cnt:] if args.history_cnt else []
            local_messages = list(local_messages)
            local_messages.append({"role": "user", "content": prompt})

            new_prompt = tokenizer.apply_chat_template(
                local_messages,
                tokenize=False,
                add_generation_prompt=True
            )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

            with torch.no_grad():
                x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)
                # DDP 下为避免多份流式输出，这里使用非流式生成
                outs = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_seq_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                answer = tokenizer.decode(outs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True)
            local_results.append((idx, prompt, answer))

        if use_ddp:
            gathered = [None for _ in range(world_size)]
            dist.gather_object(local_results, gathered if rank == 0 else None, dst=0)
            if rank == 0:
                merged = []
                for chunk in gathered:
                    if chunk:
                        merged.extend(chunk)
                merged.sort(key=lambda t: t[0])
                for _, p, a in merged:
                    print(f"Human: {p}")
                    print(f"Robot: {a}\n")
        else:
            for _, p, a in local_results:
                print(f"Human: {p}")
                print(f"Robot: {a}\n")
    else:
        # 手动输入：仅 rank0 交互与打印，其它 rank 直接返回
        if use_ddp and rank != 0:
            if dist.is_initialized():
                dist.barrier()
            return
        for idx, prompt in enumerate(iter(lambda: input('Human: '), '')):
            setup_seed(random.randint(0, 2048))
            messages = messages[-args.history_cnt:] if args.history_cnt else []
            messages.append({"role": "user", "content": prompt})

            new_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

            answer = new_prompt
            with torch.no_grad():
                x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)
                outputs = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_seq_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=args.stream,
                    pad_token_id=tokenizer.pad_token_id
                )

                print('Robot: ', end='')
                try:
                    if not args.stream:
                        print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                    else:
                        history_idx = 0
                        for y in outputs:
                            answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                            if (answer and answer[-1] == '�') or not answer:
                                continue
                            print(answer[history_idx:], end='', flush=True)
                            history_idx = len(answer)
                except StopIteration:
                    print("No answer")
                print('\n')

            messages.append({"role": "assistant", "content": answer})

    if args.ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
