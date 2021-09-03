from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import logging
import timeit
import bagua.torch_api as bagua


torch.set_printoptions(precision=20)
parser = argparse.ArgumentParser(description="alltoall benchmark test")
parser.add_argument(
    "--times-to-repeat", type=int, default=100, help="times to repeat"
)
parser.add_argument(
    "--size-of-tensor", type=int, default=100, help="size of tensor"
)
args = parser.parse_args()
assert bagua.get_world_size() >= 2, "world size must be at least 2"

torch.cuda.set_device(bagua.get_local_rank())
bagua.init_process_group()

send_tensors = [torch.rand(args.size_of_tensor, dtype=torch.float32).cuda() for i in range(bagua.get_world_size())]
recv_tensors = [torch.zeros(args.size_of_tensor, dtype=torch.float32).cuda() for i in range(bagua.get_world_size())]
recv_tensor_bagua = torch.zeros(args.size_of_tensor * bagua.get_world_size(), dtype=torch.float32).cuda()

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
if bagua.get_rank() == 0:
    logging.getLogger().setLevel(logging.INFO)

comm = bagua.get_backend("bagua_alltoall_test").global_communicator

def alltoall_comm():
    # alltoall
    dist.all_to_all(recv_tensors, send_tensors)

time = timeit.timeit(alltoall_comm, number=args.times_to_repeat)
logging.info(
        "WorldSize: %d, SizeOfTensor: %d, Repeated: %d, time: %.4f " % (
            bagua.get_world_size(), args.size_of_tensor,
            args.times_to_repeat, time)
)
#bagua.alltoall(torch.cat(send_tensors), recv_tensor_bagua, comm=comm)
#assert torch.equal(torch.cat(recv_tensors), recv_tensor_bagua),\
#    "recv_tensors:{a}, recv_tensor_bagua:{b}".format(a=recv_tensors, b=recv_tensor_bagua)


