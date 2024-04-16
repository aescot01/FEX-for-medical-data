"""A module with NAS controller-related code."""
import torch
import torch.nn.functional as F
import numpy as np
import tools
import scipy
from utils import Logger, mkdir_p
import os
import torch.nn as nn
from computational_tree import BinaryTree
import function as func
import argparse
import random
import math
import cProfile
import pstats
import scipy as sp
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import csv
# import torchdiffeq as ode
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.special import jn
from scipy.signal import find_peaks

import sys
import functools
print = functools.partial(print,flush=True)

from scipy.integrate import odeint,quad

profile = cProfile.Profile()

parser = argparse.ArgumentParser('Heat Diffusion Dynamic Case')


parser.add_argument('--sampled_time',type=str,choices=['irregular','equal'],default='irregular')
parser.add_argument('--viz',action='store_true')
parser.add_argument('--adjoint',action='store_true')
parser.add_argument('--n',type=int,default=400,help='Number of nodes')
parser.add_argument('--sparse',action='store_true')
parser.add_argument('--network',type=str,choices=['grid','random','power_law','small_world','community'],default='grid')
parser.add_argument('--layout',type=str,choices=['community','degree'],default='community')
parser.add_argument('--seed',type=int,default=0,help='Random Seed')
parser.add_argument('--stop',type=float,default=100.,help='Terminal Time')
parser.add_argument('--stop_test',type=float,default=200.,help='Terminal Time test')
parser.add_argument('--operator',type=str,choices=['lap','norm_lap','kipf','norm_adj'],default='norm_lap')
parser.add_argument('--dump',action='store_true',help='Save Results')

parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--greedy', default=0, type=float)
parser.add_argument('--random_step', default=0, type=float)
parser.add_argument('--ckpt', default='results', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dim', default=3, type=int)
parser.add_argument('--tree', default='depth1', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--percentile', default=0.5, type=float)
parser.add_argument('--dsr', default=0.5, type=float)
parser.add_argument('--base', default=100, type=int)
parser.add_argument('--domainbs', default=1000, type=int)
parser.add_argument('--bdbs', default=1000, type=int)
# argument for committor function

parser.add_argument('--Nepoch', type=int ,default = 5000)
# parser.add_argument('--bs', type=int ,default = 3000)
# parser.add_argument('--lr', type=float, default = 0.002)
parser.add_argument('--finetune',type=int,default=20000)

parser.add_argument('--lr_schedule', default='cos', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str



def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        #print("Connected to a GPU")
    else:
        #print("Using the CPU")
        device = torch.device('cpu')
    return device


# Set a random seed for reproducibility
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
# Generate 100x5 matrix X with random values
#X = np.random.rand(100, 5)
T = 100
d = 5
dt = 1
g = 1/5
N = 1000
I_hat0 = 1
S_hat0 = N-I_hat0

SIR_params=[dt,g,N,I_hat0,S_hat0,T,d]


DS = np.random.rand(T,1)

X = np.random.rand(T,d)
#

#Generate 100x1 vector y based on the formula y = sin(x1) * cos(x2)
y = np.sin(X[:, 0]) * np.cos(X[:, 1])

#If you want to reshape y to have a single column
y = y.reshape(-1, 1)





class candidate(object):
    def __init__(self, action, expression, error):
        self.action = action
        self.expression = expression
        self.error = error

class SaveBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.candidates = []

    def num_candidates(self):
        return len(self.candidates)

    def add_new(self, candidate):
        flag = 1
        action_idx = None
        for idx, old_candidate in enumerate(self.candidates):
            if candidate.action == old_candidate.action and candidate.error < old_candidate.error:  # 如果判断出来和之前的action一样的话，就不去做
                flag = 1
                action_idx = idx
                break
            elif candidate.action == old_candidate.action:
                flag = 0

        if flag == 1:
            if action_idx is not None:
                # print(action_idx)
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            self.candidates = sorted(self.candidates, key=lambda x: x.error)  # from small to large

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1)  # remove the last one

if args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

# L = 1?
elif args.tree == 'depth1':
    def basic_tree():

        tree = BinaryTree('', False)
        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

# L = 2?
elif args.tree == 'depth2_rml':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', True)

        return tree

# L = 4?
elif args.tree == 'depth2_rmu':
    print('**************************rmu**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', False)
        tree.rightChild.insertLeft('', True)
        tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth2_rmu2':
    print('**************************rmu2**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

# L = 3
elif args.tree == 'depth2_sub':
    print('**************************sub**************************')
    def basic_tree():
        tree = BinaryTree('', True)

        tree.insertLeft('', False)
        tree.leftChild.insertLeft('', True)
        tree.leftChild.insertRight('', True)

        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth4_sub':
    def basic_tree():
        tree = BinaryTree('',True)
        tree.insertLeft('',True)
        tree.leftChild.insertLeft('',False)
        tree.leftChild.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.insertRight('',True)

        return tree

elif args.tree == 'depth5':
    def basic_tree():
        tree = BinaryTree('',True)

        tree.insertLeft('',False)

        tree.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.insertLeft('',False)
        tree.leftChild.leftChild.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.leftChild.insertRight('',True)

        tree.leftChild.insertRight('',True)
        tree.leftChild.rightChild.insertLeft('',False)
        tree.leftChild.rightChild.leftChild.insertLeft('',True)
        tree.leftChild.rightChild.leftChild.insertRight('',True)

        return tree
    
structure = []

def inorder_structure(tree):
    global structure
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        inorder_structure(tree.rightChild)
inorder_structure(basic_tree())
print('tree structure', structure) # [True, False, True, True]

# 9:{0,1,Id,()^2,()^3,()^4,exp,sin,cos}, 3: {+,-,*}
structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))


print(args.tree)
if args.tree == 'depth1':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

elif args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

structure = []
leaves_index = []
leaves = 0
count = 0

def inorder_structure(tree):
    global structure, leaves, count, leaves_index
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        if tree.leftChild is None and tree.rightChild is None:
            leaves = leaves + 1
            leaves_index.append(count)
        count = count + 1
        inorder_structure(tree.rightChild)


inorder_structure(basic_tree())


print('leaves index:', leaves_index)

print('tree structure:', structure, 'leaves num:', leaves)

structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))
print('tree structure choices', structure_choice)

def reset_params(tree_params):
    for v in tree_params:
        # v.data.fill_(0.01)
        v.data.normal_(0.0, 0.1)

def inorder(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary[action]
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = binary[action]
            # print(count, action, func.binary_functions_str[action])
        count = count + 1
        inorder(tree.rightChild, actions)

def inorder_visualize(tree, actions, trainable_tree,dim):
    # actions: [tensor([12]), tensor([2]), tensor([7]), tensor([1])]
    global count, leaves_cnt
    if tree:
        leftfun = inorder_visualize(tree.leftChild, actions, trainable_tree,dim)
        action = actions[count].item() # an integer

        
        if tree.is_unary:# and not tree.key.is_leave:
            if count not in leaves_index:
                midfun = unary_functions_str[action] #e.g., '({}*({})**4+{})'
                a = trainable_tree.learnable_operator_set[count][action].a.item()
                b = trainable_tree.learnable_operator_set[count][action].b.item()
            else:
                midfun = unary_functions_str_leaf[action]
        else:
            midfun = binary_functions_str[action]
        count = count + 1
        rightfun = inorder_visualize(tree.rightChild, actions, trainable_tree,dim)

        # 若 已经在叶子结点处，得到表达式
        if leftfun is None and rightfun is None:
            w = []
            for i in range(dim):
                w.append(trainable_tree.linear[leaves_cnt].weight[0][i].item())
                # w2 = trainable_tree.linear[leaves_cnt].weight[0][1].item()
            bias = trainable_tree.linear[leaves_cnt].bias[0].item()
            leaves_cnt = leaves_cnt + 1
            ## -------------------------------------- input variable element wise  ----------------------------
            expression = ''
            for i in range(0, dim):
               # print('mid fun',midfun) #(({})**4)
                x_expression = midfun.format('x'+str(i)) # e.g., (({x2})**4)
                expression = expression + ('{:.4f}*{}'+'+').format(w[i], x_expression)
            expression = expression+'{:.4f}'.format(bias)
            expression = '('+expression+')'
            return expression

        # 当 0 或者 1 在midfun 中时，由下面例子可知midfun.format 只需2个参数，反之则需要3个参数
        # unary_functions_str examples
        # '({}*(0)+{})',
        # '({}*(1)+{})',
        # # '5',
        # '({}*{}+{})',
        # # '-{}',
        # '({}*({})**2+{})',
        # '({}*({})**3+{})',

        elif leftfun is not None and rightfun is None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), leftfun, '{:.4f}'.format(b))
        elif tree.leftChild is None and tree.rightChild is not None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), rightfun, '{:.4f}'.format(b))
        else:
            return midfun.format(leftfun, rightfun)
    else:
        return None

def get_function(actions):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder(computation_tree, actions)
    count = 0 # 置零
    return computation_tree

def inorder_params(tree, actions, unary_choices):
    global count
    if tree:
        inorder_params(tree.leftChild, actions, unary_choices)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary_choices[count][action]
            # if tree.leftChild is None and tree.rightChild is None:
            #     print('inorder_params:', count, action)
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = unary_choices[count][len(unary)+action]
            # print(count, action, func.binary_functions_str[action], tree.key(torch.tensor([1]), torch.tensor([2])))
        count = count + 1
        inorder_params(tree.rightChild, actions, unary_choices)

def get_function_trainable_params(actions, unary_choices):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder_params(computation_tree, actions, unary_choices)
    count = 0 # 置零
    return computation_tree

class unary_operation(nn.Module):
    def __init__(self, operator, is_leave):
        super(unary_operation, self).__init__()
        self.unary = operator
        if not is_leave:
            self.a = nn.Parameter(torch.Tensor(1).to(get_device()))
            self.a.data.fill_(1)
            self.b = nn.Parameter(torch.Tensor(1).to(get_device()))
            self.b.data.fill_(0)
        self.is_leave = is_leave

    def forward(self, x):
        if self.is_leave:
            return self.unary(x)
        else:
            return self.a*self.unary(x)+self.b

class binary_operation(nn.Module):
    def __init__(self, operator):
        super(binary_operation, self).__init__()
        self.binary = operator

    def forward(self, x, y):
        return self.binary(x, y)

leaves_cnt = 0

def compute_by_tree(tree, linear, x):
    ''' judge whether a emtpy tree, if yes, that means the leaves and call the unary operation '''
    if tree.leftChild == None and tree.rightChild == None: # leaf node
        global leaves_cnt
        transformation = linear[leaves_cnt]
        leaves_cnt = leaves_cnt + 1
        return transformation(tree.key(x))
    elif tree.leftChild is None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, linear, x))
    elif tree.leftChild is not None and tree.rightChild is None:
        return tree.key(compute_by_tree(tree.leftChild, linear, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, linear, x), compute_by_tree(tree.rightChild, linear, x))

class learnable_computation_tree(nn.Module):
    def __init__(self,dim):
        super(learnable_computation_tree, self).__init__()
        self.dim = dim
        self.learnable_operator_set = {}
        for i in range(len(structure)):
            self.learnable_operator_set[i] = []
            is_leave = i in leaves_index
            for j in range(len(unary)):
                self.learnable_operator_set[i].append(unary_operation(unary[j], is_leave))
            for j in range(len(binary)):
                self.learnable_operator_set[i].append(binary_operation(binary[j]))
        self.linear = []
        for num, i in enumerate(range(leaves)):
            linear_module = torch.nn.Linear(self.dim, 1, bias=True).to(get_device()) #set only one variable
            linear_module.weight.data.normal_(0, 1/math.sqrt(self.dim))
            # linear_module.weight.data[0, num%2] = 1
            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)

    def forward(self, x, bs_action):
        # print(len(bs_action))
        global leaves_cnt
        leaves_cnt = 0
        function = lambda y: compute_by_tree(get_function_trainable_params(bs_action, self.learnable_operator_set), self.linear, y)
        out = function(x)
        leaves_cnt = 0
        return out


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True

        self.input_size = 20
        self.hidden_size = 50
        self.output_size = sum(structure_choice) # e.g., 39, i.e., sum of [12, 3, 12, 12]

        self._fc_controller = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size))

    def forward(self,x):
        logits = self._fc_controller(x)

        logits /= self.softmax_temperature

        # exploration # ??
        if self.mode == 'train':
            logits = (self.tanh_c*F.tanh(logits))

        return logits

    def sample(self, batch_size=1, step=0):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        # [B, L, H]
        inputs = torch.zeros(batch_size, self.input_size).to(get_device())
        log_probs = []
        actions = []
        total_logits = self.forward(inputs) # total_logits of size (10,39)


        cumsum = np.cumsum([0]+structure_choice) # [ 0 12 15 27 39]

        for idx in range(len(structure_choice)): # structure_choice : [12 3 12 12], so idx 0,1,2,3
            logits = total_logits[:, cumsum[idx]:cumsum[idx+1]]
            probs = F.softmax(logits, dim=-1) # tensor of size 10 * 12

            # log prob of sampling a particular operator for a particular node of tree
            log_prob = F.log_softmax(logits, dim=-1) # tensor of size 10 * 12
            # print(probs)
            if step>=args.random_step:
                action = probs.multinomial(num_samples=1).data
            else:
                action = torch.randint(0, structure_choice[idx], size=(batch_size, 1)).to(get_device())

            # action is a tensor of size (batch_size,1),i.e., 对batch 中每个树的 第idx 节点，选择一个operator
            # 对batch 中每一个树，按照概率 sample 其中一个节点的action，最后action 大小为（batch_size,1)

            # 以epsilon的概率均匀sample,i.e., batch 中所有树的第idx 节点都选择同样的均匀sample出的节点
            if args.greedy != 0:
                for k in range(args.bs):
                    if np.random.rand(1)<args.greedy:
                        choice = random.choices(range(structure_choice[idx]), k=1)
                        action[k] = choice[0]
            # selected_log_prob of size (batch_size, 1)
            selected_log_prob = log_prob.gather(
                1, tools.get_variable(action, requires_grad=False))

            log_probs.append(selected_log_prob[:, 0:1])
            actions.append(action[:, 0:1])



        log_probs = torch.cat(log_probs, dim=1)   # (batch_size,number of nodes),e.g. (10,4)
        # actions的大小为（10，4）
        # print(actions)
        return actions, log_probs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))
    



def get_reward(dy,xnew,bs, actions, learnable_tree,\
             tree_params,params):
    
    dt,g,N,I_hat0,S_hat0,T,d = params

    S_hat0 = torch.tensor([S_hat0], dtype=torch.float32)  # Convert to tensor and reshape if necessary
    I_hat0 = torch.tensor([I_hat0], dtype=torch.float32)

    # Now detach them if they need to be detached
    S_hat0 = S_hat0.detach()
    I_hat0 = I_hat0.detach()  
    dy = torch.from_numpy(dy).to(get_device()).to(torch.float32)
    
    xnew = torch.from_numpy(xnew).to(get_device()).to(torch.float32)
    xnew.requires_grad = True

    
    criterion = nn.MSELoss()
    # print(x)
    regression_errors = []
    formulas = []


    batch_size = bs  # 10

    global count, leaves_cnt

    # 做T1+T2的意义是，即使给定了一组🌲的operator set，仍需要学习每个节点的weight and bias,i.e., trainable parameters

    for bs_idx in range(batch_size):
        # bs_action 是 一个batch 中完整一棵树的4个节点的分别的action
        ## keep tree1 and tree2 the same format
        bs_action = [v[bs_idx] for v in actions]


        reset_params(tree_params)



        tree_optim = torch.optim.Adam(tree_params, lr=0.005)

        
        for _ in range(T1):
            #change this later
            #copy-past lines 865-873 in the other four change locations
            #b_fex is learnable tree, xnew is the input of dimension T x # of features
            Beta = learnable_tree(xnew,bs_action)
            T = Beta.size(0)
            S_hat,I_hat = torch.zeros(T,1), torch.zeros(T,1)
            S_hat[0], I_hat[0] = S_hat0,I_hat0

            for t in range(T-1):
                hat_dSdt = -Beta[t] * S_hat[t] * I_hat[t]/N
                hat_dIdt = Beta[t] * S_hat[t] * I_hat[t]/N - g * I_hat[t]

                S_hat[t+1] = (S_hat[t] + hat_dSdt * dt).detach()
                I_hat[t+1] = (I_hat[t] + hat_dIdt * dt).detach()


            #double check the dimension of DS_hat,DS make sure they are torch.size() = (T,1)
            DS_hat = Beta * S_hat * I_hat/N


            loss = criterion(DS_hat,dy)

            tree_optim.zero_grad()
            loss.backward()

            tree_optim.step()



        tree_optim = torch.optim.LBFGS(tree_params, lr=1, max_iter=T2)


        print('---------------------------------- batch idx {} -------------------------------------'.format(bs_idx))


        error_hist = []

        def closure():
            profile.enable()
            tree_optim.zero_grad()
            #change this later
            Beta = learnable_tree(xnew,bs_action)
            T = Beta.size(0)
            S_hat,I_hat = torch.zeros(T,1), torch.zeros(T,1)
            S_hat[0], I_hat[0] = S_hat0,I_hat0

            for t in range(T-1):
                hat_dSdt = -Beta[t] * S_hat[t] * I_hat[t]/N
                hat_dIdt = Beta[t] * S_hat[t] * I_hat[t]/N - g * I_hat[t]

                S_hat[t+1] = (S_hat[t] + hat_dSdt * dt).detach()
                I_hat[t+1] = (I_hat[t] + hat_dIdt * dt).detach()


            #double check the dimension of DS_hat,DS make sure they are torch.size() = (T,1)
            DS_hat = Beta * S_hat * I_hat/N

            loss = criterion(DS_hat,dy)

    

            # print('loss before: ', loss.item())
            error_hist.append(loss.item())
            loss.backward()
            return loss

        tree_optim.step(closure)

        Beta = learnable_tree(xnew,bs_action)
        T = Beta.size(0)
        S_hat,I_hat = torch.zeros(T,1), torch.zeros(T,1)
        S_hat[0], I_hat[0] = S_hat0,I_hat0

        for t in range(T-1):
            hat_dSdt = -Beta[t] * S_hat[t] * I_hat[t]/N
            hat_dIdt = Beta[t] * S_hat[t] * I_hat[t]/N - g * I_hat[t]

            S_hat[t+1] = (S_hat[t] + hat_dSdt * dt).detach()
            I_hat[t+1] = (I_hat[t] + hat_dIdt * dt).detach()


        #double check the dimension of DS_hat,DS make sure they are torch.size() = (T,1)
        DS_hat = Beta * S_hat * I_hat/N


        loss = criterion(DS_hat,dy)


        regression_error = loss


        error_hist.append(regression_error.item())

        print(' min: ', min(error_hist))
        regression_errors.append(min(error_hist))

        count = 0
        leaves_cnt = 0
        formula = inorder_visualize(basic_tree(), bs_action, learnable_tree,dim=5)
        count = 0
        leaves_cnt = 0




        formulas.append(formula)




    return regression_errors, formulas

    

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def true(x):
    return -0.5*(torch.sum(x**2, dim=1, keepdim=True))

def best_error(xnew,dy,best_action,\
                learnable_tree,params):

    dt,g,N,I_hat0,S_hat0,T,d = params
    S_hat0 = torch.tensor([S_hat0], dtype=torch.float32)  # Convert to tensor and reshape if necessary
    I_hat0 = torch.tensor([I_hat0], dtype=torch.float32)

    # Now detach them if they need to be detached
    S_hat0 = S_hat0.detach()
    I_hat0 = I_hat0.detach()  
    criterion = nn.MSELoss()

    dy = torch.from_numpy(dy).to(get_device()).to(torch.float32)
    
    xnew = torch.from_numpy(xnew).to(get_device()).to(torch.float32)
    xnew.requires_grad = True

    # keep tree1 and tree2 same format
    bs_action = best_action


    Beta = learnable_tree(xnew,bs_action)
    T = Beta.size(0)
    S_hat,I_hat = torch.zeros(T,1), torch.zeros(T,1)
    S_hat[0], I_hat[0] = S_hat0,I_hat0

    for t in range(T-1):
        hat_dSdt = -Beta[t] * S_hat[t] * I_hat[t]/N
        hat_dIdt = Beta[t] * S_hat[t] * I_hat[t]/N - g * I_hat[t]

        S_hat[t+1] = (S_hat[t] + hat_dSdt * dt).detach()
        I_hat[t+1] = (I_hat[t] + hat_dIdt * dt).detach()


    #double check the dimension of DS_hat,DS make sure they are torch.size() = (T,1)
    DS_hat = Beta * S_hat * I_hat/N


    loss = criterion(DS_hat,dy)


    regression_error = loss

    return regression_error


def train_controller(xtest,dytest,controller, controller_optim,\
     learnable_tree, tree_params,\
          hyperparams,params):

    ### obtain a new file name ###
    dt,g,N,I_hat0,S_hat0,T,d = params


    file_name = os.path.join(hyperparams['checkpoint'], 'tree_{}_log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(4,file_idx)):
        file_idx += 1
    file_name = file_name.format(4,file_idx)
    logger_tree = Logger(file_name, title='')
    logger_tree.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])




    model4 = controller




    model4.train()




    baseline = None

    bs = args.bs
    smallest_error = float('inf')


    pool_size = 10

    candidates = SaveBuffer(pool_size)



    for step in range(hyperparams['controller_max_step']):
        # sample models
        actions, log_probs = controller.sample(batch_size=bs, step=step)


        binary_code = ''
        for action in actions:
            binary_code = binary_code + str(action[0].item())






        # 这时的rewards 实际上是error, formulas是 batch 里分别的数学表达式

        rewards,formulas = get_reward(DS,X,bs, actions, learnable_tree,\
                                      tree_params,SIR_params)



        rewards = torch.FloatTensor(rewards).to(get_device()).view(-1,1) # torch.Size([10])

        # discount
        if 1 > hyperparams['discount'] > 0:
            rewards = discount(rewards, hyperparams['discount'])

        base = args.base
        rewards[rewards > base] = base
        rewards[rewards != rewards] = 1e10
        error = rewards # size (bs,1)

        # 此时的rewards 为真正的rewards，i.e., 越大越好
        rewards = 1 / (1 + torch.sqrt(rewards))

        batch_smallest = error.min()
        batch_min_idx = torch.argmin(error)


        batch_min_action = [v[batch_min_idx] for v in actions]


        batch_best_formula = formulas[batch_min_idx]

        # print('action',batch_min_action)
        # print('expression',batch_best_formula)
        # print('error,',batch_smallest)

        candidates.add_new(candidate(action=batch_min_action, expression=batch_best_formula, error=batch_smallest))

        for candidate_ in candidates.candidates:
            print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action], candidate_.expression))


        
        # moving average baseline
        if baseline is None:
            baseline = (rewards).mean()
        else:
            decay = hyperparams['ema_baseline_decay']
            baseline = decay * baseline + (1 - decay) * (rewards).mean()


        argsort = torch.argsort(rewards.squeeze(1), descending=True)  # [6, 5, 2, 3, 4, 0, 1, 7, 8, 9]

        # print(error, argsort)
        # print(rewards.size(), rewards.squeeze(1), torch.argsort(rewards.squeeze(1)), rewards[argsort])
        # policy loss
        num = int(args.bs * args.percentile)
        rewards_sort = rewards[argsort]
        adv = rewards_sort - rewards_sort[num:num + 1, 0:]  # - baseline
        # print(error, argsort, rewards_sort, adv)

  # (bs, len(tree))
        log_probs_sort = log_probs[argsort]

        # print('adv', adv)
        ################### Loss in the form of expectation ######################
        loss = -(log_probs_sort[:num]) * tools.get_variable(adv[:num], False, requires_grad=False)
        loss = (loss.sum(1)).mean()

        # update


        controller_optim.zero_grad()

        loss.backward()

        if hyperparams['controller_grad_clip'] > 0:


            torch.nn.utils.clip_grad_norm_(model4.parameters(),
                                          hyperparams['controller_grad_clip'])

        controller_optim.step()

        min_error = error.min().item()
        # print('******************** ', min_error)
        if smallest_error > min_error:
            smallest_error = min_error

            min_idx = torch.argmin(error)

            min_action = [v[min_idx] for v in actions]
            best_formula = formulas[min_idx]




        log = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
                'Reward {re:.4f} | {error:.8f} {formula}'.format(loss=loss.item(), base=baseline, act=binary_code,
                                                    re=(rewards).mean(), step=step, formula=best_formula,
                                                    error=smallest_error)
        
        print('********************************************************************************************************')
        print(log)
        print('********************************************************************************************************')
        if (step + 1) % 1 == 0:
            logger_tree.append([step + 1, loss.item(), baseline, rewards.mean(), smallest_error, best_formula])




    for candidate_ in candidates.candidates:
        print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action],
                                                    candidate_.expression))
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger_tree.append([666, 0, 0, action_string, candidate_.error.item(), candidate_.expression])





        # logger.append([666, 0, 0, 0, candidate_.error.item(), candidate_.expression]) 


    ############## Finetune candidates in the pool
    finetune = args.finetune

    global count, leaves_cnt
    for candidate_ in candidates.candidates:
        trainable_tree = learnable_computation_tree(dim=5)
        trainable_tree = trainable_tree.to(get_device())

        params = []
        for idx, v in enumerate(trainable_tree.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params.append(param)
        for module in trainable_tree.linear:
            for param in module.parameters():
                params.append(param)





        reset_params(params)

        tree_optim = torch.optim.Adam(params, lr=1e-2)


        
        for current_iter in range(finetune):
            error = best_error(X,DS,candidate_.action,\
                                trainable_tree,SIR_params)

            tree_optim.zero_grad()

            error.backward()

            if hyperparams['finetune_grad_clip'] > 0:

                torch.nn.utils.clip_grad_norm_(params,
                    max_norm = hyperparams['finetune_grad_clip'])




            tree_optim.step()



            count = 0
            leaves_cnt = 0


            formula = inorder_visualize(basic_tree(),candidate_.action,trainable_tree,dim=5)
            leaves_cnt = 0
            count = 0



            suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=current_iter, error=error, formula=formula)
            if (current_iter + 1) % 100 == 0:
                logger_tree.append([current_iter, 0, 0, 0, error.item(), formula])

            if args.lr_schedule == 'cos':
                cosine_lr(tree_optim, 1e-2, current_iter, finetune)
            elif args.lr_schedule == 'exp':
                expo_lr(tree_optim,1e-2,current_iter,gamma=0.999)



        if isinstance(xtest, np.ndarray):
            xtest = torch.from_numpy(xtest).to(get_device()).float()
        if isinstance(dytest,np.ndarray):
            dytest = torch.from_numpy(dytest).to(get_device()).float()

        # ✅

        criterion = nn.MSELoss()
        Beta = learnable_tree(xtest,candidate_.action)
        T = Beta.size(0)
        S_hat,I_hat = torch.zeros(T,1), torch.zeros(T,1)
        S_hat[0], I_hat[0] = S_hat0,I_hat0

        for t in range(T-1):
            hat_dSdt = -Beta[t] * S_hat[t] * I_hat[t]/N
            hat_dIdt = Beta[t] * S_hat[t] * I_hat[t]/N - g * I_hat[t]

            S_hat[t+1] = (S_hat[t] + hat_dSdt * dt).detach()
            I_hat[t+1] = (I_hat[t] + hat_dIdt * dt).detach()


        #double check the dimension of DS_hat,DS make sure they are torch.size() = (T,1)
        DS_hat = Beta * S_hat * I_hat/N


        loss = criterion(DS_hat,dytest)        

        print('l2 error: ', loss.item())

        logger_tree.append(['l2 error', 0, 0, 0, loss.item(), 0])



def expo_lr(opt,base_lr,e,gamma):
    lr = base_lr * gamma ** e
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':

    controller = Controller().to(get_device())


    hyperparams = {}

    hyperparams['controller_max_step'] = args.epoch
    hyperparams['discount'] = 1.0
    hyperparams['ema_baseline_decay'] = 0.95
    hyperparams['controller_lr'] = args.lr
    hyperparams['entropy_mode'] = 'reward'
    hyperparams['controller_grad_clip'] = 0 #10
    hyperparams['finetune_grad_clip'] = 0
    hyperparams['checkpoint'] = args.ckpt

    if not os.path.isdir(hyperparams['checkpoint']):
        print(hyperparams['checkpoint'])
        mkdir_p(hyperparams['checkpoint'])


    controller_optim = torch.optim.Adam(controller.parameters(), lr= hyperparams['controller_lr'])

    #change dim according to input data
    trainable_tree = learnable_computation_tree(dim=5).to(get_device())



    


    params = []
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in trainable_tree.linear:
        for param in module.parameters():
            params.append(param)




    T1 = 20
    T2 = 20


    train_controller(X,y,controller, controller_optim,\
    trainable_tree, params, hyperparams,SIR_params)
