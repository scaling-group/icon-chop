import torch
from einops import repeat, rearrange

'''
u: [usize, udim]
lb: [udim]
rb: [udim]
out: [usize+2, udim]
'''

def roll_db_2(u, lb, rb):
    """Dirichlet boundary for roll = 2"""
    return torch.cat([repeat(lb, 'd -> 3 d'), u[:-1, :]], dim=0)

def roll_db_1(u, lb, rb):
    """Dirichlet boundary for roll = 1"""
    return torch.cat([repeat(lb, 'd -> 2 d'), u], dim=0)

def roll_db_0(u, lb, rb):
    """Dirichlet boundary for roll = 0"""
    return torch.cat([lb[None, :], u, rb[None, :]], dim=0)

def roll_db_neg1(u, lb, rb):
    """Dirichlet boundary for roll = -1"""
    return torch.cat([u, repeat(rb, 'd -> 2 d')], dim=0)

def roll_db_neg2(u, lb, rb):
    """Dirichlet boundary for roll = -2"""
    return torch.cat([u[1:, :], repeat(rb, 'd -> 3 d')], dim=0)

roll_db_funs = {
    2: roll_db_2,
    1: roll_db_1,
    0: roll_db_0,
    -1: roll_db_neg1,
    -2: roll_db_neg2,
}

def roll_pb_2(u):
    """Periodic boundary for roll = 2"""
    return torch.cat([u[-3:, :], u[:-1, :]], dim=0)

def roll_pb_1(u):
    """Periodic boundary for roll = 1"""
    return torch.cat([u[-2:, :], u], dim=0)

def roll_pb_0(u):
    """Periodic boundary for roll = 0"""
    return torch.cat([u[-1:, :], u, u[:1, :]], dim=0)

def roll_pb_neg1(u):
    """Periodic boundary for roll = -1"""
    return torch.cat([u, u[:2, :]], dim=0)

def roll_pb_neg2(u):
    """Periodic boundary for roll = -2"""
    return torch.cat([u[1:, :], u[:3, :]], dim=0)

roll_pb_funs = {
    2: roll_pb_2,
    1: roll_pb_1,
    0: roll_pb_0,
    -1: roll_pb_neg1,
    -2: roll_pb_neg2,
}

def get_u_roll_dirichlet(u, left_bound, right_bound, roll_list):
    '''
    u: [usize, udim]
    u_roll: [usize + 2, udim, 5]
    '''
    rolls = []
    for j in roll_list:
        rolled = roll_db_funs[j](u, left_bound, right_bound)
        rolls.append(rolled)
    
    # Stack along last dimension using einops for clarity
    u_roll = torch.stack(rolls, dim=-1).to(torch.float32)  # stencil for u_{i}
    return u_roll

def get_u_roll_periodic(u, left_bound, right_bound, roll_list):
    '''
    u: [usize, udim]
    u_roll: [usize + 2, udim, 5]
    boundaries are dummy
    '''
    u = u.transpose(0,1)
    rolls = []
    for j in roll_list:
        rolled = roll_pb_funs[j](u)
        rolls.append(rolled)
    
    # Stack along last dimension using einops for clarity
    u_roll = torch.stack(rolls, dim=-1).to(torch.float32)  # stencil for u_{i}
    return u_roll.transpose(0,1)

if __name__ == "__main__":
    u = torch.stack([torch.arange(10), torch.arange(10)*2], dim=1).float()
    lb = torch.tensor([-99.0, -100.0])
    rb = torch.tensor([99.0, 100.0])
    
    r1 = get_u_roll_dirichlet(u, lb, rb, (2, 1, 0, -1, -2))
    print(r1[:, 0, :].T)
    print(r1[:, 1, :].T)
    print('----------')
    
    r2 = get_u_roll_periodic(u, lb, rb, (2, 1, 0, -1, -2))
    print(r2[:, 0, :].T)
    print(r2[:, 1, :].T)
