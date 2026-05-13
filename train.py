#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import time


running = True
reserved_tensors = []


def handle_signal(signum, frame):
    global running
    running = False
    print("\n收到停止信号，训练结束后会释放 GPU 显存。", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run continuous synthetic training on selected GPUs.")
    parser.add_argument("--gpu-ids", default="4,5,6", help="物理 GPU 编号，默认 4,5,6。")
    parser.add_argument("--steps", type=int, default=0, help="训练步数；0 表示一直训练。")
    parser.add_argument("--batch-size", type=int, default=128, help="每张 GPU 的 batch size。")
    parser.add_argument("--hidden-size", type=int, default=4096, help="MLP hidden size。")
    parser.add_argument("--layers", type=int, default=40, help="MLP 线性层数量。")
    parser.add_argument("--lr", type=float, default=1e-3, help="SGD 学习率。")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="训练 dtype；auto 会优先使用 bfloat16，否则 float16。",
    )
    parser.add_argument(
        "--reserve-mem-fraction",
        type=float,
        default=0.90,
        help="训练预热后每张 GPU 尽量保留到的显存占用比例；0 表示不额外占显存。",
    )
    parser.add_argument("--reserve-chunk-mb", type=int, default=256, help="额外占显存的分块大小。")
    parser.add_argument("--safety-mb", type=int, default=1024, help="每张 GPU 至少留下的空闲显存。")
    parser.add_argument("--log-interval", type=int, default=10, help="每多少步打印一次日志。")
    return parser.parse_args()


def import_torch_after_cuda_visible_devices(args):
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境没有 torch，请先激活安装了 PyTorch 的训练环境。") from exc

    return torch


def resolve_dtype(torch, dtype_name):
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_model(torch, hidden_size, layers):
    modules = []
    for _ in range(layers):
        modules.append(torch.nn.Linear(hidden_size, hidden_size, bias=False))
        modules.append(torch.nn.GELU())
    modules.append(torch.nn.Linear(hidden_size, hidden_size, bias=False))
    return torch.nn.Sequential(*modules)


class GpuTrainer:
    def __init__(self, torch, device_id, args, dtype):
        self.torch = torch
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.model = build_model(torch, args.hidden_size, args.layers).to(self.device, dtype=dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        self.inputs = torch.randn(args.batch_size, args.hidden_size, device=self.device, dtype=dtype)
        self.targets = torch.randn(args.batch_size, args.hidden_size, device=self.device, dtype=dtype)

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(self.inputs)
        loss = (outputs.float() - self.targets.float()).pow(2).mean()
        loss.backward()
        self.optimizer.step()
        return loss.detach()


def reserve_memory(torch, args):
    if args.reserve_mem_fraction <= 0:
        return
    if not 0 < args.reserve_mem_fraction < 1:
        raise ValueError("--reserve-mem-fraction 需要在 0 和 1 之间，或设为 0 关闭额外占显存。")

    safety_bytes = args.safety_mb * 1024 * 1024
    chunk_bytes = args.reserve_chunk_mb * 1024 * 1024

    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            target_free_bytes = int(total_bytes * (1 - args.reserve_mem_fraction)) + safety_bytes
            bytes_to_reserve = max(0, free_bytes - target_free_bytes)
            device_tensors = []
            reserved_bytes = 0

            while reserved_bytes < bytes_to_reserve:
                request_bytes = min(chunk_bytes, bytes_to_reserve - reserved_bytes)
                request_elements = request_bytes
                try:
                    device_tensors.append(torch.empty(request_elements, dtype=torch.uint8, device="cuda"))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                reserved_bytes += request_bytes

            reserved_tensors.append(device_tensors)
            print(
                f"GPU {device_id}: 额外保留显存约 {reserved_bytes / 1024**3:.2f} GiB",
                flush=True,
            )


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    torch = import_torch_after_cuda_visible_devices(args)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda 不可用，请确认当前环境安装了 CUDA 版本的 PyTorch。")

    dtype = resolve_dtype(torch, args.dtype)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError(f"没有识别到可见 GPU，请检查 CUDA_VISIBLE_DEVICES={visible_devices} 是否正确。")

    print(f"CUDA_VISIBLE_DEVICES={visible_devices}", flush=True)
    print(f"可见 GPU 数量: {device_count}; dtype={dtype}", flush=True)

    trainers = []
    for device_id in range(device_count):
        props = torch.cuda.get_device_properties(device_id)
        print(f"GPU {device_id}: {props.name}", flush=True)
        trainers.append(GpuTrainer(torch, device_id, args, dtype))

    warmup_loss = [trainer.step() for trainer in trainers]
    for device_id in range(device_count):
        torch.cuda.synchronize(device_id)
    print(
        "预热完成: "
        + ", ".join(f"gpu{idx}_loss={loss.item():.6f}" for idx, loss in enumerate(warmup_loss)),
        flush=True,
    )

    reserve_memory(torch, args)
    print("开始持续训练。按 Ctrl+C 或 kill 进程即可停止。", flush=True)

    step = 0
    last_losses = warmup_loss
    start_time = time.time()

    while running and (args.steps <= 0 or step < args.steps):
        last_losses = [trainer.step() for trainer in trainers]
        step += 1

        if step % args.log_interval == 0:
            for device_id in range(device_count):
                torch.cuda.synchronize(device_id)
            elapsed = max(time.time() - start_time, 1e-6)
            losses = ", ".join(f"gpu{idx}_loss={loss.item():.6f}" for idx, loss in enumerate(last_losses))
            print(f"step={step} speed={step / elapsed:.2f} steps/s {losses}", flush=True)

    for device_id in range(device_count):
        torch.cuda.synchronize(device_id)
    print(f"训练结束，总步数: {step}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        sys.exit(1)
