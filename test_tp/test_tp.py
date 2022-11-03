import torch
from torch import nn
import torch.distributed as dist
import os
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1= MyLinear(10, 6)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MyLinear(nn.Linear):
    def __init__(
        self,
        inputfeatures,
        outputfeatures,
        bias=True
    ):
        super(MyLinear, self).__init__(
            inputfeatures, outputfeatures, bias
        )
        self.rank = dist.get_rank()
        self.device = torch.device(f'cuda:{self.rank}')
        self.world_size = dist.get_world_size()
    
    def forward(self, x):
        print(self.weight)
        tmp_out = torch.matmul(x, self.weight.t())
        print(tmp_out)
        tensor_list = [torch.zeros_like(tmp_out) for _ in range(self.world_size)]
        # out_tensor = torch.zeros_like(torch.cat(tensor_list, dim=0), device=self.device)
        # print(tmp_out.shape)
        dist.all_gather(tensor_list, tmp_out)
        # print(self.rank, "\n", tensor_list)
        out = torch.cat(tensor_list, dim=1)
        print(out.shape)
        return out

class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= MyLinear2(10, 6)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MyLinear2(nn.Linear):
    def __init__(
        self,
        inputfeatures,
        outputfeatures,
        bias=True
    ):
        super(MyLinear2, self).__init__(
            inputfeatures, outputfeatures, bias
        )
           
    def forward(self, x):
        out = torch.matmul(x, self.weight.t())
        return out


def run_tp():
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    outputpath = 'tp.csv'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model = MLP().to(rank)
    for module in model.children():
        weight_list = list(module.weight.chunk(world_size, dim=0))
        weight = weight_list[rank]
        module.weight = nn.Parameter(weight)

    x = torch.randn((6, 10), device=torch.device(f'cuda:{rank}'))
    dist.broadcast(x, src=0)
    out = model(x).detach().cpu().numpy()
    if rank == 0:
        np.savetxt(outputpath, out, fmt='%.2f',delimiter=',')

def run_ori():
    outputpath = 'ori.csv'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP2().to(DEVICE)
    x = torch.randn((6, 10)).to(DEVICE)
    out = model(x).detach().cpu().numpy()
    np.savetxt(outputpath, out, fmt='%.2f',delimiter=',')


if __name__ == '__main__':
    seed = 10086
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    run_tp()
