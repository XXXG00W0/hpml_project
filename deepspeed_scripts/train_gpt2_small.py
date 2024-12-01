import os
import math
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DDP and Deepspeed with ZeRO optimization for GPT-2 training')

    # Data and model paths
    parser.add_argument('--data-path', type=str, default='../oscar_subsets/oscar_subset_100MB_raw.jsonl', help='Path to data jsonl file')
    parser.add_argument('--vocab-file', type=str, default='../gpt2_small/vocab.json', help='Path to vocab.json')
    parser.add_argument('--merge-file', type=str, default='../gpt2_small/merges.txt', help='Path to merges.txt')
    parser.add_argument('--checkpoint-path', type=str, default='/checkpoints/', help='Path to save checkpoints')
    parser.add_argument('--tensorboard-logs-path', type=str, default='/logs/', help='TensorBoard log directory')

    # Training parameters
    parser.add_argument('--micro-batch-size', type=int, default=8, help='Micro batch size per GPU')
    parser.add_argument('--global-batch-size', type=int, default=64, help='Global batch size across all GPUs')
    parser.add_argument('--train-iters', type=int, default=500, help='Total number of training iterations')
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

    # Logging and checkpointing
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=1000, help='Checkpoint saving interval')
    parser.add_argument('--wandb-project', type=str, default='megatron-gpt2-benchmark', help='WandB project name')

    # Distributed training
    parser.add_argument('--distributed-framework', type=str, default=None, help='Distributed framework to use (deepspeed, torch or None)')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs per node')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master-port', type=int, default=6000, help='Master port for distributed training')
    parser.add_argument('--use-pytorch-profiler', action='store_true', help='Enable PyTorch Profiler')
    parser.add_argument('--profile-ranks', type=str, default='0', help='Comma-separated ranks to profile')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--init-method-std', type=float, default=0.02, help='Standard deviation for model initialization')

    # Not implemented arguments
    parser.add_argument('--world-size', type=int, default=None, help='World size for distributed training')
    parser.add_argument('--data-impl', type=str, default='mmap', help='Data implementation type (mmap, lazy, etc.)')
    parser.add_argument('--split', type=str, default='949,50,1', help='Data split ratios (train, validation, test)')
    parser.add_argument('--empty-unused-memory-level', type=int, default=1, help='Level for emptying unused memory (1-3)')
    parser.add_argument('--log-throughput', action='store_true', help='Enable logging of throughput')
    parser.add_argument('--timing-log-level', type=int, default=2, help='Timing log level (0-3)')
    parser.add_argument('--timing-log-option', type=str, default='all', help='Timing log option (default: all)')
    parser.add_argument('--log-timers-to-tensorboard', action='store_true', help='Log timers to TensorBoard')
    parser.add_argument('--log-validation-ppl-to-tensorboard', action='store_true', help='Log validation perplexity to TensorBoard')
    parser.add_argument('--cuda-max-connections', type=int, default=1, help='Set CUDA_DEVICE_MAX_CONNECTIONS')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

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
    train_dataset = TextDataset(tokenizer, train_lines, block_size=1024)
    valid_dataset = TextDataset(tokenizer, valid_lines, block_size=1024)
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

def evaluate(model_engine, valid_dataloader, eval_iters):
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            if i >= eval_iters:
                break
            input_ids = batch['input_ids'].to(model_engine.local_rank)
            labels = batch['labels'].to(model_engine.local_rank)
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    model_engine.train()
    return avg_loss, perplexity

def setup_deepspeed(model, args):
    
    deepspeed_config = {
        "train_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {
            "enabled": args.use_fp16
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [args.adam_beta1, args.adam_beta2],
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupCosine",
            "params": {
                "warmup_min_lr": args.min_lr,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": warmup_num_steps,
                "total_num_steps": args.train_iters
            }
        },
        "gradient_clipping": args.clip_grad,
        "zero_optimization": {
            "stage": 2,  # ZeRO Stage 2 for optimizer state partitioning
            "offload_optimizer": {
                "device": "cpu",  # Offload optimizer states to CPU
                "pin_memory": True
            },
            "overlap_comm": True
        }
    }
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=deepspeed_config
    )

    world_size = args.num_gpus * args.num_nodes
    gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    warmup_num_steps = min(2000, int(args.train_iters * 0.1))
    return model_engine, optimizer

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

def setup_ddp(model, args):
    torch.distributed.init_process_group(backend='nccl')
    model = model.to('cuda')
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  betas=(args.adam_beta1, args.adam_beta2), 
                                  weight_decay=args.weight_decay)
    # imitate DeepSpeed's WarmupCosine schedule
    scheduler = create_warmup_cosine_schedule(args, optimizer)
    scalar = torch.amp.GradScaler() if args.use_fp16 else None
    return model, scheduler, scalar

def setup_no_distributed(model, args):
    model = model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  betas=(args.adam_beta1, args.adam_beta2), 
                                  weight_decay=args.weight_decay)
    scheduler = create_warmup_cosine_schedule(args, optimizer)
    scalar = torch.amp.GradScaler() if args.use_fp16 else None
    return model, scheduler, scalar

def train_step_deepspeed(model_engine, batch):
    input_ids = batch['input_ids'].to(model_engine.local_rank)
    labels = batch['labels'].to(model_engine.local_rank)
    outputs = model_engine(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
    return loss

def train_step_torch(model_engine, batch, optimizer, scaler, args):

    input_ids = batch['input_ids'].to(model_engine.device)
    labels = batch['labels'].to(model_engine.device)

    with torch.cuda.amp.autocast(device_type='cuda', enabled=args.use_fp16, dtype=torch.float16):
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

    model_engine.zero_grad()
    scaler.scale(loss).backward()

    if args.clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.clip_grad)

    scaler.step(optimizer)
    scaler.update()
    return loss

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file)
    tokenizer.pad_token = tokenizer.eos_token

    # Use pretrained GPT-2 Small
    model = GPT2LMHeadModel.from_pretrained('gpt2-small')
    model.config.initializer_range = args.init_method_std  # From INITIALIZATION_ARGS

    # Calculate gradient_accumulation_steps
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    # Set up DeepSpeed configuration
    total_training_steps = args.train_iters
    args.warmup_num_steps = min(2000, int(total_training_steps * 0.1))

    # Set Distriburted Framework
    model_engine, optimizer, scalar = None, None, None
    if args.distributed_framework == 'deepspeed':
        model_engine, optimizer = setup_deepspeed(model, args)
    elif args.distributed_framework == 'torch':
        model_engine, optimizer, scaler = setup_ddp(model, args)
    elif args.distributed_framework is None:
        model_engine, optimizer, scaler = setup_no_distributed(model, args)
    else:
        raise ValueError(f"Invalid distributed framework: {args.distributed_framework}")

    # Load datasets
    train_dataset, valid_dataset = create_datasets(tokenizer, args.data_path, train_val_split=args.train_val_split)

    # Create data loaders
    train_dataloader, valid_dataloader = create_train_val_dataloader(train_dataset, valid_dataset, args.micro_batch_size)

    # Set up logging
    writer = SummaryWriter(log_dir=args.tensorboard_logs_path)
    if args.distributed_framework == 'deepspeed':
        wandb.init(project=args.wandb_project+' deepspeed', dir='./wandb')
    elif args.distributed_framework == 'torch':
        wandb.init(project=args.wandb_project+' torch', dir='./wandb')
    else:
        wandb.init(project=args.wandb_project, dir='./wandb')
    wandb.config.update(args)

    # PyTorch Profiler
    profiler_ranks = [int(rank) for rank in args.profile_ranks.split(',')]
    profiler = None
    if args.use_pytorch_profiler and model_engine.local_rank in profiler_ranks:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=5, active=10, repeat=10),
            record_shapes=False, 
            profile_memory=True, 
            with_stack=False, 
            use_cuda=True, 
            tensorboard_trace_handler=tensorboard_trace_handler(args.tensorboard_logs_path))
        profiler.start()

    # Training loop
    global_step = 0
    while global_step < args.train_iters:
        for batch in train_dataloader:
            if global_step >= args.train_iters:
                break
            
            if args.distributed_framework == 'deepspeed':
                loss = train_step_deepspeed(model_engine, batch)
            elif args.distributed_framework == 'torch' or args.distributed_framework is None:
                loss = train_step_torch(model_engine, batch, optimizer, scalar, args)

            if args.use_pytorch_profiler and model_engine.local_rank in profiler_ranks:
                profiler.step()
                print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                wandb.log({'profiler': profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10)}, step=global_step)

            if global_step % args.log_interval == 0:
                wandb.log({'loss': loss.item()}, step=global_step)
                writer.add_scalar('Loss/train', loss.item(), global_step)
                print(f"Step {global_step}: loss {loss.item()}")

            if global_step % args.eval_interval == 0 and global_step != 0:
                avg_loss, perplexity = evaluate(model_engine, valid_dataloader, args.eval_iters)
                wandb.log({'validation_loss': avg_loss, 'perplexity': perplexity}, step=global_step)
                writer.add_scalar('Loss/valid', avg_loss, global_step)
                writer.add_scalar('Perplexity/valid', perplexity, global_step)
                print(f"Validation loss: {avg_loss}, Perplexity: {perplexity}")

            if global_step % args.save_interval == 0 and global_step != 0:
                save_path = os.path.join(args.checkpoint_path, f'checkpoint-{global_step}')
                model_engine.save_checkpoint(save_path)
                print(f"Checkpoint saved at {save_path}")

            global_step += 1
    
    # Stop PyTorch Profiler
    if args.use_pytorch_profiler and model_engine.local_rank in profiler_ranks:
        profiler.stop()
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Final evaluation
    avg_loss, perplexity = evaluate(model_engine, valid_dataloader, args.eval_iters)
    wandb.log({'validation_loss': avg_loss, 'perplexity': perplexity}, step=global_step)
    writer.add_scalar('Loss/valid', avg_loss, global_step)
    writer.add_scalar('Perplexity/valid', perplexity, global_step)
    print(f"Final Validation loss: {avg_loss}, Perplexity: {perplexity}")

    # Save final model
    save_path = os.path.join(args.checkpoint_path, f'checkpoint-{global_step}')
    model_engine.save_checkpoint(save_path)
    print(f"Final checkpoint saved at {save_path}")

    writer.close()
    wandb.finish()

if __name__ == '__main__':
    main()
