import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models

# import horovod.torch as hvd
import timeit
import numpy as np

# 导入bagua包
import bagua.torch_api as bagua


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Synthetic Benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)

parser.add_argument("--model", type=str, default="resnet50", help="model to benchmark")
parser.add_argument("--batch-size", type=int, default=32, help="input batch size")

parser.add_argument(
    "--num-warmup-batches",
    type=int,
    default=10,
    help="number of warm-up batches that don't count towards benchmark",
)
parser.add_argument(
    "--num-batches-per-iter",
    type=int,
    default=10,
    help="number of batches per benchmark iteration",
)
parser.add_argument(
    "--num-iters", type=int, default=10, help="number of benchmark iterations"
)

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

parser.add_argument(
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction",
)

# bagua args
parser.add_argument(
    "--algorithm",
    type=str,
    default="allreduce",
    help="allreduce, quantize, decentralize or 1bit-adam",
)
parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="use amp",
    )

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 初始化分布式运行环境
torch.cuda.set_device(bagua.get_local_rank())

bagua.init_process_group()

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

# hvd.init()

# if args.cuda:
#     # Horovod: pin GPU to local rank.
#     torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = bagua.get_world_size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # # If using GPU Adasum allreduce, scale learning rate by local_size.
    # if args.use_adasum and hvd.nccl_built():
    #     lr_scaler = hvd.local_size()

optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

# 用Bagua模型、优化器覆盖原始模型和优化器
model, optimizer = bagua.bagua_init(model, optimizer, distributed_algorithm=args.algorithm)

# # Horovod: (optional) compression algorithm.
# compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# # Horovod: wrap optimizer with DistributedOptimizer.
# optimizer = hvd.DistributedOptimizer(optimizer,
#                                      named_parameters=model.named_parameters(),
#                                      compression=compression,
#                                      op=hvd.Adasum if args.use_adasum else hvd.Average)

# # Horovod: broadcast parameters & optimizer state.
# hvd.broadcast_parameters(model.state_dict(), root_rank=0)
# hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()


def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)

    if args.amp:
        scaled_loss = args.scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        return

    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if bagua.get_rank() != 0:
        return
    print(s, end="\n" if nl else "")


log("Model: %s" % args.model)
log("Batch size: %d" % args.batch_size)
device = "GPU" if args.cuda else "CPU"
log("Number of %ss: %d" % (device, bagua.get_world_size()))

# Warm-up
log("Running warmup...")
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log("Running benchmark...")
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log("Iter #%d: %.1f img/sec %s" % (x, img_sec * bagua.get_world_size(), device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log("Img/sec per %s: %.1f +-%.1f" % (device, img_sec_mean, img_sec_conf))
log(
    "Total img/sec on %d %s(s): %.1f +-%.1f"
    % (
        bagua.get_world_size(),
        device,
        bagua.get_world_size() * img_sec_mean,
        bagua.get_world_size() * img_sec_conf,
    )
)
