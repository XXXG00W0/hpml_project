import os, gc, sys
import math
import argparse
import time
import torch
import ctypes
import GPUtil
import tabulate
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import pandas as pd
import wandb
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from tqdm import tqdm

# Megatron-LM Timers implementation
from timers import Timers
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DDP and Deepspeed with ZeRO optimization for GPT-2 training')

    # Data and model paths
    parser.add_argument('--data-path', type=str, default='./oscar_subsets/oscar_subset_100MB_raw.jsonl', help='Path to data jsonl file')
    parser.add_argument('--model-path', type=str, default='./gpt2_small', help='Path to GPT-2 model')
    parser.add_argument('--vocab-file', type=str, default='./gpt2_small/vocab.json', help='Path to vocab.json')
    parser.add_argument('--merge-file', type=str, default='./gpt2_small/merges.txt', help='Path to merges.txt')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--tensorboard-logs-path', type=str, default='./tb_logs', help='TensorBoard log directory')

    # Model parameters
    parser.add_argument('--kv-channels', type=int, default=64, help='Key/Value channels')
    parser.add_argument('--num-attention-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--hidden-size', type=int, default=768, help='Hidden size')
    parser.add_argument('--group-query-attention', action='store_true', help='Use group query attention')
    parser.add_argument('--num-query-groups', type=int, default=12, help='Number of query groups')
    parser.add_argument('--num-experts', type=int, default=None, help='Number of experts')
    parser.add_argument('--moe-router-topk', type=int, default=1, help='MoE router top-k')
    parser.add_argument('--swiglu', action='store_true', help='Use SwiGLU activation function')
    parser.add_argument('--moe-shared-expert-intermediate-size', type=int, default=None, help='MoE shared expert intermediate size')
    parser.add_argument('--seq-length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--ffn-hidden-size', type=int, default=3072, help='FFN hidden size')
    parser.add_argument('--padded-vocab-size', type=int, default=50257, help='Padded vocab size')

    # Training parameters
    parser.add_argument('--micro-batch-size', type=int, default=8, help='Micro batch size per GPU')
    parser.add_argument('--global-batch-size', type=int, default=32, help='Global batch size across all GPUs')
    parser.add_argument('--train-iters', type=int, default=30, help='500 Total number of training iterations')
    parser.add_argument('--train-val-split', type=float, default=0.9, help='Train-Validation split ratio')
    parser.add_argument('--eval-iters', type=int, default=10, help='Number of iterations for evaluation')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.00015, help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--clip-grad', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--lr-decay-style', type=str, default='cosine', help='Learning rate decay style')
    parser.add_argument('--warmup-num-steps', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--adam-beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam-beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--use-fp16', action='store_true', help='Use FP16 mixed precision training')
    
    # Memory management
    parser.add_argument('--empty-unused-memory-level', type=int, default=1, help='Level for emptying unused memory (1-3)')
    parser.add_argument('--clear-memory-interval', type=int, default=5, help='100 Interval for clearing unused memory')

    # Logging and checkpointing
    parser.add_argument('--log-interval', type=int, default=10, help='100 Logging interval')
    parser.add_argument('--eval-interval', type=int, default=10, help='1000 Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=10, help='1000 Checkpoint saving interval')
    parser.add_argument('--wandb-project', type=str, default='gpt2-benchmark', help='WandB project name')
    parser.add_argument('--timing-log-level', type=int, default=0, choices=range(0,3), help='Timing log level (0-3)')
    parser.add_argument('--timing-log-option', type=str, default='minmax', choices=['max', 'minmax', 'all'], 
                        help='Timing log option (max, minmax, all)')
    parser.add_argument('--log-throughput', action='store_true', help='Enable logging of throughput')
    parser.add_argument('--log-timers-to-tensorboard', action='store_true', help='Log timers to TensorBoard')

    # Distributed training
    parser.add_argument('--framework', type=str, default=None, help='Distributed framework to use (deepspeed, torch or None)')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs per node')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master-port', type=int, default=6000, help='Master port for distributed training')
    parser.add_argument('--use-pytorch-profiler', action='store_true', help='Enable PyTorch Profiler')
    parser.add_argument('--profile-ranks', type=str, default='0', help='Comma-separated ranks to profile')
    parser.add_argument('--world-size', type=int, default=None, help='World size for distributed training')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--init-method-std', type=float, default=0.02, help='Standard deviation for model initialization')

    # Not implemented arguments
    parser.add_argument('--data-impl', type=str, default='mmap', help='Data implementation type (mmap, lazy, etc.)')
    parser.add_argument('--split', type=str, default='949,50,1', help='Data split ratios (train, validation, test)')
    parser.add_argument('--log-validation-ppl-to-tensorboard', action='store_true', help='Log validation perplexity to TensorBoard')
    parser.add_argument('--cuda-max-connections', type=int, default=1, help='Set CUDA_DEVICE_MAX_CONNECTIONS')

    args = parser.parse_args()
    return args

def check_directory(args):
    chpt_run = f'{args.checkpoint_path}/run_{time.strftime("%Y%m%d-%H%M%S")}'   
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        # Create a new directory for each run
        os.makedirs(os.path.join(args.checkpoint_path, chpt_run))
    if not os.path.exists(args.tensorboard_logs_path):
        os.makedirs(args.tensorboard_logs_path)

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=1024):
        assert os.path.isfile(file_path)
        with open(file_path, encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.examples = tokenizer(lines, return_tensors='pt', max_length=block_size, truncation=True, padding='max_length')['input_ids']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {'input_ids': self.examples[i], 'labels': self.examples[i]}

def create_datasets(tokenizer, data_path, train_val_split=0.9):
    '''
    Create train and validation datasets from a single txt file.
    '''
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    split_idx = int(len(lines) * train_val_split)
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]
    
    data_dir = os.path.dirname(data_path)
    train_path = os.path.join(data_dir, 'train.txt')
    valid_path = os.path.join(data_dir, 'valid.txt')
    if not os.path.exists(train_path):
        with open(train_path, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
    if not os.path.exists(valid_path):
        with open(valid_path, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
    train_dataset = TextDataset(tokenizer, train_path, block_size=1024)
    valid_dataset = TextDataset(tokenizer, valid_path, block_size=1024)
    return train_dataset, valid_dataset

def create_train_val_dataloader(train_dataset, valid_dataset, micro_batch_size):
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=micro_batch_size,
        num_workers=4,
        pin_memory=True
    )
    return train_dataloader, valid_dataloader

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_list = []
    for gpu in gpus:
        gpu_info = {
            "ID": gpu.id,
            "Name": gpu.name,
            "Load (%)": f"{gpu.load * 100:.1f}%",
            "Free Memory (MB)": f"{gpu.memoryFree:.1f}",
            "Used Memory (MB)": f"{gpu.memoryUsed:.1f}",
            "Total Memory (MB)": f"{gpu.memoryTotal:.1f}",
            "Temperature (°C)": f"{gpu.temperature}°C"
        }
        gpu_list.append(gpu_info)
    
    return tabulate(gpu_list, headers="keys", tablefmt="pretty")

def evaluate(model_engine, valid_dataloader, eval_iters, args):
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(total=eval_iters, desc='Validation')
    
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            if i >= eval_iters:
                break
                
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            total_steps += 1
            pbar.update(1)
    pbar.close()

    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_steps
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        print("Perplexity is too large to calculate")
        perplexity = float('inf')
    model_engine.train()
    return avg_loss, perplexity

def create_warmup_cosine_schedule(args, optimizer):
    warmup_num_steps = args.warmup_num_steps
    def warmup_schedule(current_step):
        if current_step < warmup_num_steps:
            return current_step / warmup_num_steps
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.train_iters - warmup_num_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_num_steps])
    return scheduler

def setup_no_distributed(model, args):
    model = model.to('cuda')
    if args.framework == 'dp':
        model = DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  betas=(args.adam_beta1, args.adam_beta2), 
                                  weight_decay=args.weight_decay)
    scheduler = create_warmup_cosine_schedule(args, optimizer)
    scalar = torch.amp.GradScaler() if args.use_fp16 else None
    return model, optimizer, scheduler, scalar

def train_step_torch(model_engine, batch, optimizer, scheduler, scaler, args, global_step):

    input_ids = batch['input_ids'].to(model_engine.device)
    labels = batch['labels'].to(model_engine.device)

    with torch.autocast(device_type='cuda', enabled=args.use_fp16, dtype=torch.float16):
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

    # Empty unused memory.
    if global_step % args.clear_memory_interval == 0 \
    and global_step != 0 \
    and args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    model_engine.zero_grad()
    if args.use_fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if args.clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.clip_grad)

    if args.use_fp16:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    scheduler.step()

    # Empty unused memory.
    if global_step % args.clear_memory_interval == 0 \
    and global_step != 0 \
    and args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    return loss

def main():
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    args = parse_args()
    check_directory(args)
    torch.manual_seed(args.seed)    

    if args.world_size is None or args.world_size != args.num_gpus * args.num_nodes:
        args.world_size = args.num_gpus * args.num_nodes
    
    # Set up logging
    # log_dir = os.path.join(cwd, f'{args.tensorboard_logs_path}/run_{time.strftime("%Y%m%d-%H%M%S")}')
    log_dir = args.tensorboard_logs_path
    print(f"TensorBoard logs path: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file)
    tokenizer.pad_token = tokenizer.eos_token

    # Use pretrained GPT-2 Small
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.config.initializer_range = args.init_method_std  # From INITIALIZATION_ARGS

    # Calculate gradient_accumulation_steps
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    # Set up DeepSpeed configuration
    total_training_steps = args.train_iters
    args.warmup_num_steps = min(2000, int(total_training_steps * 0.1))

    # Set Distriburted Framework
    model_engine, optimizer, scheduler, scalar = setup_no_distributed(model, args)

    # Load datasets
    train_dataset, valid_dataset = create_datasets(tokenizer, args.data_path, train_val_split=args.train_val_split)

    # Create data loaders
    train_dataloader, valid_dataloader = create_train_val_dataloader(train_dataset, valid_dataset, args.micro_batch_size)

    # Initialize WandB offline
    wandb_enabled = False
    try:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)
        wandb_enabled = True
    except Exception as e:
        print(f"Error initializing WandB: {e}")

    # PyTorch Profiler
    profiler_ranks = [int(rank) for rank in args.profile_ranks.split(',')]
    profiler = None
    if args.use_pytorch_profiler:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=5, active=15, repeat=1),
            record_shapes=True, 
            profile_memory=True, 
            with_stack=False,
            on_trace_ready=tensorboard_trace_handler(args.tensorboard_logs_path))
        profiler.start()
        print(f"Profiler initialized")

    global_step = 0
    pbar = tqdm(total=args.train_iters, desc='Training')

    # Timers
    timers = Timers(args.timing_log_level, args.timing_log_option)
    # Distribute training require barrier before starting the timer
    timers('interval-time', log_level=0).start(barrier=args.framework is not None)

    # Training loop
    while global_step < args.train_iters:
        for batch in train_dataloader:
            if global_step >= args.train_iters:
                break
            
            
            loss = train_step_torch(model_engine, batch, optimizer, scheduler, scalar, args, global_step)

            # Profiler
            if args.use_pytorch_profiler:
                profiler.step()

            # WandB and TensorBoard logging
            if global_step % args.log_interval == 0 and global_step != 0:
                elapsed_time = timers('interval-time').stop(barrier=args.framework is not None)
                elapsed_time = timers('interval-time').elapsed(barrier=args.framework is not None)

                # Log loss
                if wandb_enabled: wandb.log({'loss': loss.item()}, step=global_step)
                writer.add_scalar('Loss/train', loss.item(), global_step)
                print(f"Step {global_step}: loss {loss.item()}")

                # Log learning rate
                learning_rate = scheduler.get_last_lr()[0]
                if wandb_enabled: wandb.log({'learning-rate': learning_rate}, step=global_step)
                writer.add_scalar('Learning-rate', learning_rate, global_step)
                print(f"Learning rate: {learning_rate}")

                # Log throughput
                if args.log_throughput:
                    elapsed_time_per_iteration = elapsed_time / global_step
                    batch_size = args.micro_batch_size * args.world_size

                    throughput = num_floating_point_operations(args, batch_size) / (
                        elapsed_time_per_iteration * 10**12 * args.world_size)
                    if wandb_enabled: wandb.log({'throughput': throughput}, step=global_step)
                    writer.add_scalar('Throughput', throughput, global_step)
                    print(f"Throughput over steps {global_step}: {throughput}")
                # Log loss scale
                # to-do: log loss scale

                # Log memory usage
                mem_usage = torch.cuda.memory_allocated() / (1024 ** 2)
                if wandb_enabled: wandb.log({'memory-usage': mem_usage}, step=global_step)
                writer.add_scalar('Memory-usage', mem_usage, global_step)
                print(f"Torch reports memory usage: {mem_usage} MB")
                print(f"GPUtil reports memory usage: \n{get_gpu_info()}")

                # Log time to tensorboard
                if args.log_timers_to_tensorboard:
                    writer.add_scalar('iteration-time', elapsed_time_per_iteration, global_step)
                    if wandb_enabled: wandb.log({'iteration-time': elapsed_time_per_iteration}, global_step)

                # Resume training timer
                timers('interval-time', log_level=0).start(barrier=args.framework is not None)

            # Validation
            if global_step % args.eval_interval == 0 and global_step != 0:
                # Stop training timer and start interval timer
                timers('interval-time').stop()
                timers('eval-time', log_level=0).start(barrier=args.framework is not None)
                avg_loss, perplexity = evaluate(model_engine, valid_dataloader, args.eval_iters, args)
                timers('eval-time').stop()
                if wandb_enabled: wandb.log({'validation_loss': avg_loss, 'perplexity': perplexity}, step=global_step)
                writer.add_scalar('Loss/valid', avg_loss, global_step)
                writer.add_scalar('Perplexity/valid', perplexity, global_step)
                print(f"Validation loss: {avg_loss}, Perplexity: {perplexity}")
                # Resume training timer
                timers('interval-time', log_level=0).start(barrier=args.framework is not None)

            # Save checkpoint
            if global_step % args.save_interval == 0 and global_step != 0:
                save_path = os.path.join(args.checkpoint_path, f'checkpoint-{global_step}')
                if args.framework == 'deepspeed':
                    model_engine.save_checkpoint(save_path)
                else:
                    torch.save(model_engine.state_dict(), save_path)
                print(f"Checkpoint saved at {save_path}")

            global_step += 1
            pbar.update(1)

    pbar.close()
    
    # Stop PyTorch Profiler and log data
    if args.use_pytorch_profiler:
        key_averages = profiler.key_averages()
        
        # print top 10 ops by self_cuda_time_total
        print(key_averages.table(sort_by="self_cuda_time_total", row_limit=10))
        
        # accumulate the average times for ops that start with "torch::mm"
        total_device_time = sum(item.device_time_total for item in key_averages)
        total_cpu_time = sum(item.cpu_time_total for item in key_averages)

        # Prepare data for logging
        data = [
            {
                "Name": item.key,
                "Device Time Total (ms)": item.device_time_total / 1e3,
                "CPU Time Total (ms)": item.cpu_time_total / 1e3,
                "Self Device Time Total (ms)": item.self_device_time_total / 1e3,
                "Self CPU Time Total (ms)": item.self_cpu_time_total / 1e3,
                "Calls": item.count,
                "Input Shapes": item.input_shapes,
                "Device Memory Usage (MB)": item.device_memory_usage / (1024 ** 2),
                "Self Device Memory Usage (MB)": item.self_device_memory_usage / (1024 ** 2),
                "CPU Memory Usage (MB)": item.cpu_memory_usage / (1024 ** 2),
                "Self CPU Memory Usage (MB)": item.self_cpu_memory_usage / (1024 ** 2),
            }
            for item in key_averages
        ]

        # Print total device and CPU time
        print(f"Total Device Time: {total_device_time / 1e3:.2f} ms")
        print(f"Total CPU Time: {total_cpu_time / 1e3:.2f} ms")
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(data)

        # Log data to WandB and TensorBoard
        if wandb_enabled:
            wandb.log({'profiler_data': wandb.Table(dataframe=df)}, step=global_step)
            wandb.log({'total_device_time_ms': total_device_time / 1e3}, step=global_step)
            wandb.log({'total_cpu_time_ms': total_cpu_time / 1e3}, step=global_step)

        # import time
        # profiler.export_chrome_trace(f'{args.tensorboard_logs_path}/trace_{time.strftime("%Y%m%d-%H%M%S")}.json')
        
        profiler.stop()

    # Final evaluation
    avg_loss, perplexity = evaluate(model_engine, valid_dataloader, args.eval_iters, args)
    if wandb_enabled: wandb.log({'validation_loss': avg_loss, 'perplexity': perplexity}, step=global_step)
    writer.add_scalar('Loss/valid', avg_loss, global_step)
    writer.add_scalar('Perplexity/valid', perplexity, global_step)
    print(f"Final Validation loss: {avg_loss}, Perplexity: {perplexity}")

    # Save final model
    save_path = os.path.join(args.checkpoint_path, f'checkpoint-{global_step}')
    torch.save(model_engine.state_dict(), save_path)
    print(f"Final checkpoint saved at {save_path}")

    writer.close()
    if wandb_enabled: wandb.finish()

if __name__ == '__main__':
    main()