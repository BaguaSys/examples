Use the following script to start elastic training with x node * 8 gpus:

Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

```bash
python3 -m bagua.distributed.run \
        --nnodes=1:4 \
        --nproc_per_node=8 \
        --rdzv_id=JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=192.168.1.1:1234 \
        main.py
```

Node 2:

```bash
python3 -m bagua.distributed.run \
        --nnodes=1:4 \
        --nproc_per_node=8 \
        --rdzv_id=JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=192.168.1.1:1234 \
        main.py
```
