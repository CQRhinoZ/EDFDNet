# # 分布式
# import argparse
# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
#
#
#     # # 分布式训练 初始化进程组
#     args.local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(args.local_rank)
#     dist.init_process_group("gloo", init_method="env://")
#
#
#     # 分布式
#     parser.add_argument("--local_rank", type=int, default=0)
#
#
#
#     # 分布式
#     model.cuda(args.local_rank)
#     self.model = DDP(model, device_ids=[args.local_rank],output_device=args.local_rank, find_unused_parameters=True)
#     self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))