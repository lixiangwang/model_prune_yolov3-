import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def parse_module_defs(module_defs):
    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                # spp前一个CBL不剪 区分tiny
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'route' and 'groups' in module_defs[i + 1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

        elif module_def['type'] == 'upsample':
            # 上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def parse_module_defs2(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == 1:
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'route' and 'groups' in module_defs[i + 1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'upsample':
            # 上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

        elif module_def['type'] == 'shortcut':
            identity_idx = (i + int(str(module_def['from'])[1:-1]))
            if module_defs[identity_idx]['type'] == 'convolutional':

                # ignore_idx.add(identity_idx)
                shortcut_idx[i - 1] = identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':

                # ignore_idx.add(identity_idx - 1)
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all


def parse_module_defs4(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == 1:
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
        elif module_def['type'] == 'shortcut':
            shortcut_idx.append(i - 1)

    return CBL_idx, Conv_idx, shortcut_idx


def gather_bn_weights(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key == 'batch_normalize' and value == 0:
                    continue

                if key != 'type':
                    if key == 'anchors':
                        value = ', '.join(','.join(str(int(i)) for i in j) for j in value)
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


class BNOptimizer():
    '''
    稀疏训练
    mode==1: 恒定s给bn回传添加额外梯度
    mode==2: 全局s衰减，前50%epoch恒定s，后50%恒定s*alpha
    mode==3: 局部s衰减：前50%epoch恒定s，后50%中，对percent比例的通道保持s，1-percent比例的通道衰减s*alpha
    '''
    @staticmethod
    def updateBN(module_list, s, prune_idx, epoch,opt=None,mode = None,percent = 0.85, alpha = 0.01):
        if mode == 1:
            for idx in prune_idx:
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
        elif mode ==2:
            s = s if epoch <= opt.epochs * 0.5 else s * alpha
            for idx in prune_idx:
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
        elif mode == 3:
            if opt.sr and opt.prune==1 and epoch > opt.epochs * 0.5:
                idx2mask = get_mask2(module_list, prune_idx, percent)
            if idx2mask:
                for idx in idx2mask:
                    bn_module = module_list[idx][1]
                    # bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.data.sub_((1-alpha) * s * torch.sign(bn_module.weight.data) * idx2mask[idx].cuda())


def obtain_quantiles(bn_weights, num_quantile=5):
    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total // num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile + 1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):
    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers']:
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        if len(route_in_idxs) == 1:
            mask = CBLidx2mask[route_in_idxs[0]]
            if 'groups' in module_defs[idx - 1]:
                return mask[(mask.shape[0] // 2):]
            return mask

        elif len(route_in_idxs) > 1:
            tmp = []
            for i in range(len(route_in_idxs)):
                if module_defs[route_in_idxs[i]]['type'] == 'upsample':
                    mask = CBLidx2mask[route_in_idxs[i] - 1]
                elif module_defs[route_in_idxs[i]]['type'] == 'convolutional':
                    mask = CBLidx2mask[route_in_idxs[i]]
                else:
                    mask = CBLidx2mask[route_in_idxs[i] - 1]               
                tmp.append(mask)    
            return np.concatenate(tmp)
        else:
            print("Something wrong with route module!")
            raise Exception
    elif module_defs[idx - 1]['type'] == 'maxpool':  # tiny
        if module_defs[idx - 2]['type'] == 'route':  # v4 tiny
            return get_input_mask(module_defs, idx - 1, CBLidx2mask)
        else:
            return CBLidx2mask[idx - 2]  # v3 tiny


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)

        for next_idx in next_idx_list:
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):
    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_idx][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_idx in CBL_idx:
            next_bn = pruned_model.module_list[next_idx][1]
            next_bn.running_mean.data.sub_(offset)
        else:
            next_conv.bias.data.add_(offset)


def prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'][0])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)



        elif model_def['type'] == 'route':
            # spp不参与剪枝，其中的route不用更新，仅占位
#             from_layers = [int(s) for s in model_def['layers'].split(',')]
            from_layers = model_def['layers']
            activation = None
            tmp = []
            for index in range(len(from_layers)):
                actv = activations[i + from_layers[index] if from_layers[index] < 0 else from_layers[index]]
                tmp.append(actv)
                activation = torch.cat(tmp)
            update_activation(i, pruned_model, activation, CBL_idx)
        
#             if len(from_layers) == 1:
#                 activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
#                 if 'groups' in model_def:
#                     activation = activation[(activation.shape[0]//2):]
#                 update_activation(i, pruned_model, activation, CBL_idx)
#             elif len(from_layers) == 2:
#                 actv1 = activations[i + from_layers[0]]
#                 actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
#                 activation = torch.cat((actv1, actv2))
#                 update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i - 1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  # 区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

    return pruned_model


def get_mask(module_list, prune_idx, shortcut_idx):
    sort_prune_idx = [idx for idx in prune_idx if idx not in shortcut_idx]
    bn_weights = gather_bn_weights(module_list, sort_prune_idx)
    sorted_bn = torch.sort(bn_weights)[0]
    highest_thre = []
    for idx in sort_prune_idx:
        # .item()可以得到张量里的元素值
        highest_thre.append(module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)
    filters_mask = []
    idx_new = dict()
    # CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in prune_idx:
        bn_module = module_list[idx][1]
        if idx not in shortcut_idx:
            mask = obtain_bn_mask(bn_module, torch.tensor(highest_thre)).cpu()
            idx_new[idx] = mask
        else:
            mask = idx_new[shortcut_idx[idx]]
            idx_new[idx] = mask

        filters_mask.append(mask.clone())

    prune2mask = {idx: mask for idx, mask in zip(prune_idx, filters_mask)}
    return prune2mask


def get_mask2(module_list, prune_idx, percent):
    bn_weights = gather_bn_weights(module_list, prune_idx)
    print(bn_weights)
    sorted_bn = torch.sort(bn_weights)[0]
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]

    filters_mask = []
    for idx in prune_idx:
        bn_module = module_list[idx][1]
        mask = obtain_bn_mask(bn_module, thre).cpu()
        filters_mask.append(mask.clone())

    prune2mask = {idx: mask for idx, mask in zip(prune_idx, filters_mask)}
    return prune2mask


def merge_mask(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue

            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                layer_i = int(model.module_defs[layer_i]['from'][0]) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())

                layer_i = int(model.module_defs[layer_i]['from'][0]) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())
