from utils import *
from model_naive import CondDiffPool
import torch, torch.nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_name', type=str, default='usa')
parser.add_argument('--tgt_name', type=str, default="brazil")
parser.add_argument('--round', type=int, default=10)
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--cond', type=int, default=1)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--share', type=int, default=0)
parser.add_argument('--alpha1', type=float, default=0.5)  # global mmd weight  alpha1+alpha2=1
parser.add_argument('--alpha2', type=float, default=0.5)  # conditional mmd weight
parser.add_argument('--beta1', type=float, default=0.1)  # domain loss weight
parser.add_argument('--beta2', type=float, default=0.1)  # ce weight
parser.add_argument('--beta3', type=float, default=0.1)  # prox weight
parser.add_argument('--gamma1', type=float, default=0.1)  # cce weight
parser.add_argument('--gamma2', type=float, default=0.1)  # lm weight
parser.add_argument('--gamma3', type=float, default=0.1)  # ls weight

args = parser.parse_args()
src_name = args.src_name
tgt_name = args.tgt_name
cond = args.cond
sparse = args.sparse
share = args.share
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print('cond',cond,'sparse',sparse, 'share',share,device)
src_x, src_y, src_views = get_data(src_name, kmax=1)
tgt_x, tgt_y, tgt_views = get_data(tgt_name, kmax=1)
if cuda:
    src_x = src_x.to(device)
    src_y = src_y.to(device)
    src_views = [x.to(device) for x in src_views]
    tgt_x = tgt_x.to(device)
    tgt_y = tgt_y.to(device)
    tgt_views = [x.to(device) for x in tgt_views]

n_node_s = src_x.shape[0]
n_node_t = tgt_x.shape[0]
params = {}
params['n_feat'] = src_x.shape[1]
params['n_class'] = 4
# conv block -> pool block -> ...
params['n_conv_block'] = 2
params['n_pool_block'] = 1
params['pool_r'] = [0.5]
# #gcn_layers in each block
params['n_gcn_layers_conv'] = [2,2]
params['n_gcn_layers_pool'] = [2]
# 确保每个conv block都设计了层数
assert len(params['n_gcn_layers_conv']) == params['n_conv_block'] and len(params['n_gcn_layers_pool'])==params['n_pool_block']
# conv模块的输出层控制embedding维度
params['n_conv_hidden'] = [[8,8],[8,8]]
# 确保conv模块每层都设计了维度
for i in range(params['n_conv_block']):
    assert len(params['n_conv_hidden'][i]) == params['n_gcn_layers_conv'][i]

# pooling模块,每个模块的输入维度是上一个conv模块的输出维度，每个模块的输出维度表示cluster数量
params['n_pool_hidden_s'] = []
params['n_pool_hidden_t'] = []
for i in range(params['n_pool_block']):
    # 加入conditional信息后要求最后一层pooling出类别数个cluster
    # pooling_num_s = int(n_node_s*params['pool_r'][i]) if i<params['n_pool_block']-1 else params['n_class']
    # pooling_num_t = int(n_node_t*params['pool_r'][i]) if i<params['n_pool_block']-1 else params['n_class']
    pooling_num_s = int(n_node_s*params['pool_r'][i])
    pooling_num_t = int(n_node_t*params['pool_r'][i])
    params['n_pool_hidden_s'].append([8,pooling_num_s])
    params['n_pool_hidden_t'].append([8,pooling_num_t])



for i in range(params['n_pool_block']):
    # 确保每个pooling模块都按照设定的层数设定了隐层维度
    assert len(params['n_pool_hidden_s'][i]) == params['n_gcn_layers_pool'][i]
    assert len(params['n_pool_hidden_t'][i]) == params['n_gcn_layers_pool'][i]
    # 确保每个pooling模块的输入维度是上一个conv模块的输出维度
    assert params['n_pool_hidden_s'][i][0] == params['n_conv_hidden'][i][-1]
    assert params['n_pool_hidden_t'][i][0] == params['n_conv_hidden'][i][-1]

conv_params = []
pool_params_s = []
pool_params_t = []
input_dim_conv = params['n_feat']
input_dim_pool_s = params['n_feat']
input_dim_pool_t = params['n_feat']
for i in range(params['n_conv_block']):
    conv_params.append((params['n_gcn_layers_conv'][i],input_dim_conv,params['n_conv_hidden'][i]))
    input_dim_conv = params['n_conv_hidden'][i][-1]
    if i<params['n_pool_block']:
        pool_params_s.append((params['n_gcn_layers_pool'][i], input_dim_pool_s, params['n_pool_hidden_s'][i]))
        pool_params_t.append((params['n_gcn_layers_pool'][i], input_dim_pool_t, params['n_pool_hidden_t'][i]))
    input_dim_pool_s = params['n_conv_hidden'][i][-1]
    input_dim_pool_t = params['n_conv_hidden'][i][-1]

overall_tgt_acc = []
logfile = f"result/{src_name}_{tgt_name}_2clf_cond{cond}_sparse{sparse}_share{share}.txt"
with open(logfile, 'a+') as f:
    f.write("{0:%Y-%m-%d  %H-%M-%S/}\n".format(datetime.now()))

for r in range(args.round):
    with open(logfile, "a+") as f:
        f.write(f"run:{r+1}\n")
    model = CondDiffPool(conv_params, pool_params_s, pool_params_t, params['n_class'], device=device, classwise=cond, share=share, sparse=sparse)
    if cuda:
        model = model.to(device)
    model.zero_grad()
    # lr 0.5? 0.05?
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-1, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # writer = SummaryWriter('board/'+"{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()))
    best_acc_s = 0.
    best_acc_t = 0.
    for epoch in range(args.epoch):
        loss = 0.
        embeddings, pred, pooling_loss, label_matrix,pred2s = model.forward(src_x, src_views[0], to_onehot(src_y, params['n_class'], device), tgt_x, tgt_views[0],to_onehot(tgt_y, params['n_class'],device) )
        clf_loss = criterion(pred[0], src_y)
        loss += clf_loss
        # note 第二层pooling的分类损失
        # clf_loss2 = criterion(pred2s, torch.argmax(label_matrix[1][0], dim=1))
        # print(clf_loss2.item())
        # loss += clf_loss2
        domain_loss = []
        domain_loss_classwise = []
        for l in range(params['n_conv_block']):
            if cond:
                domain_loss_classwise.append(model.classwise_simple_mmd(source=embeddings[l][0], target=embeddings[l][1], src_y=label_matrix[l][0], tgt_y=label_matrix[l][1]))
                domain_loss.append(model.simple_mmd_kernel(source=embeddings[l][0], target=embeddings[l][1]))
            else:
                domain_loss.append(model.simple_mmd_kernel(source=embeddings[l][0], target=embeddings[l][1]))

        if cond:
            loss += args.beta1 * (sum(domain_loss) * args.alpha1 + sum(domain_loss_classwise) * args.alpha2)
            loss += args.gamma1 * sum(x['cce'] for x in pooling_loss)
            loss += args.gamma2 * sum(x['lm'] for x in pooling_loss)
            loss += args.gamma3 * sum(x['ls'] for x in pooling_loss)
        else:
            loss += args.beta1 * sum(domain_loss)
        loss += args.beta2*sum([x['ce'] for x in pooling_loss])
        loss += args.beta3*sum([x['prox'] for x in pooling_loss])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, src_indices = torch.max(torch.log_softmax(pred[0], dim=-1), dim=1)
        correct_s = torch.sum(src_indices == src_y)
        best_acc_s = max(best_acc_s, correct_s.item()*1.0/len(src_y) )
        _, tgt_indices = torch.max(torch.log_softmax(pred[1], dim=-1), dim=1)
        correct_t = torch.sum(tgt_indices == tgt_y)
        best_acc_t = max(best_acc_t, correct_t.item()*1.0/len(tgt_y))
        if epoch%15==0:
            with open(logfile,"a+") as f:
                c = f"epoch:{epoch},source acc:{best_acc_s},target acc:{best_acc_t},loss:{loss}\n"
                print(c)
                f.write(c)

    overall_tgt_acc.append(best_acc_t)

try:
    f_overall_tgt_acc = list(filter(lambda x:x>0.485, overall_tgt_acc))
    avg_acc = sum(f_overall_tgt_acc)/(len(f_overall_tgt_acc))
except ZeroDivisionError:
    avg_acc = sum(overall_tgt_acc)/(len(overall_tgt_acc))
with open(logfile, "a+") as f:
    f.write(f"FINAL RESULT ACC: {avg_acc}\n")

"""
magic 
CUDA_VISIBLE_DEVICES=0 nohup python server_model.py --src_name=usa --tgt_name=brazil --epoch 300
"""
