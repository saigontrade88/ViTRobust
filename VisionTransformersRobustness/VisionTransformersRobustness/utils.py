import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import Optional

def rank_print(text: str):
	"""
	Prints a statement with an indication of what node rank is sending it
	"""
	rank = dist.get_rank()
	# Keep the print statement as a one-liner to guarantee that
	# one single process prints all the lines
	print(f"Rank: {rank}, {text}.")

def setup_distrib(rank: int, world_size: int, 
				  init_method: str = "/work_bgfs/l/longdang/Distributed-Machine-Learning-with-Python/Chapter03/dp_1_mp/sharedfile",
				  backend: str = "nccl"):
	# select the correct device for this process
	torch.cuda.set_device(rank)
	# initialize the processing group with the communication backend
	torch.distributed.init_process_group(backend=backend, 
										world_size=world_size, 
										# init_method=init_method, 
										rank=rank)
	# return the current device
	return torch.device(f"cuda:{rank}")