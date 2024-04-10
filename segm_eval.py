# -*- coding: utf-8 -*-
import cv2
import torch
from einops import einops
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
import numpy as np
# from models.resnet import resnet50 as resnet
# from models import dtd_resnet as sa_opt
# from CNNs.baselines import dtd_ww as sa_base
# from CNNs.baselines import dtd_monvaton as sa_base
# from CNNs.baselines import lrp_0 as sa_base
# from CNNs.baselines import dtd_z_b as sa_base
# from CNNs.baselines import dtd_z_plus as sa_base
# from models.vit import vit_base_patch16_224 as vit_lrp
from models.vit_exp import vit_base_patch16_224 as vit_lrp_exp
from models.vit_lrp_manager import ignite_relprop
# from models.vgg_AGF import vgg19
# from models.vgg import vgg19
from data.voc import VOCSegmentation
# from data.imagenet import ImagenetSegDataset
from tqdm import tqdm
# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)


def pixel_accuracy_batch(vis, label):  # [b 224 224]
    vis = vis.cpu().numpy()
    label = label.cpu().numpy()
    label = label.reshape(label.shape[0], -1)
    vis = vis.reshape(vis.shape[0], -1)
    pixel_labeled = np.sum(label > 0)
    if pixel_labeled == 0:
        return None
    pixel_correct = np.sum((vis == label) * (label > 0))
    return pixel_correct / pixel_labeled


def iou_batch(vis, label):  # [b 224 224]
    vis = vis.cpu().numpy()
    label = label.cpu().numpy()
    label = label.reshape(label.shape[0], -1)
    vis = vis.reshape(vis.shape[0], -1)
    intersection = np.sum((vis == label) * (label > 0))
    union = np.sum((vis + label) > 0) / 2
    return intersection / union


pixel_acc = []
intersection = []


def eval(model, loader):
    iterator = tqdm(loader)
    # net = sa_base.ActivationStoringNet(sa_base.model_flattening(model)).cuda()
    # DTD = sa_base.DTD().cuda()
    # net = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model)).cuda()
    # DTD = sa_opt.DTDOpt().cuda()

    for batch_idx, (x, label) in enumerate(iterator):  # [B 3 224 224], [B 224 224]
        x = x.cuda()
        label = label.to(x.device)

        # model inference
        # module_stack, output = net(x)
        # vis = DTD(module_stack, output, 1000, 'resnet')  # [b, 3, 224, 224]
        _ = model(x)
        vis = ignite_relprop(model, input=x,  # [b 1 224 224]
                             method="transformer_attribution", alpha=1,
                             index=label.data.cpu()).detach()
        # kwargs = {
        #     'no_a': False,
        #     'no_fx': False,
        #     'no_fdx': False,
        #     'no_m': False,
        #     'no_reg': False,
        #     'gradcam': False
        # }
        # vis = model.AGF(**kwargs)

        # vivi = []
        # for v in vis:
        #     v = torch.sum(v, dim=0, keepdim=False)
        #     v = (v - v.min()) / (v.max() - v.min())  # normalize
        #     ret = v.mean()
        #     v = v.gt(ret)
        #     v = torch.where(torch.gt(v, ret), torch.ones_like(v), v)
        #     v = v.cpu().data.numpy()
        #     vivi.append(v)
        # vis = np.array(vivi)
        # vis = torch.from_numpy(vis).to(x.device)

        # vivi = []
        # for v in vis:
        #     v = torch.sum(v, dim=0)
        #     # v = gaussian_filter(v.cpu().data.numpy(), sigma=1)
        #     v = (v - v.min()) / (v.max() - v.min() + 1e-9)  # normalize
        #     v = v.cpu().data
        #     v = cv2.applyColorMap(np.uint8(255 * v), cv2.COLORMAP_TURBO)  # TURBO 线性， JET 减弱为微小值
        #     v = np.float32(v) / 255
        #     vivi.append(v)
        # vis = np.array(vivi)
        # vis = torch.from_numpy(vis).to(x.device)
        # vis = einops.rearrange(vis, 'b h w c -> b c h w')
        # vis = torch.sum(vis, dim=1)

        vis = torch.sum(vis, dim=1, keepdim=False)
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-9)
        ret = vis.mean()
        vis = vis.gt(ret)
        vis = torch.where(vis > 0, 1, 0)

        pa = pixel_accuracy_batch(vis, label)
        iou = iou_batch(vis, label)
        if pa is None:
            continue

        pixel_acc.append(pa)
        intersection.append(iou)



        iterator.set_description('PixAcc: %.4f, IoU: %.4f' % (np.array(pixel_acc).mean() * 100,
                                                              np.array(intersection).mean() * 100))

    print('[Eval Summary]:')
    print('mPA: {:.2f}%, mIoU: {:.2f}%'.format(np.array(pixel_acc).mean() * 100,
                                               np.array(intersection).mean() * 100))


if __name__ == "__main__":


    imagenet_ds = VOCSegmentation('/data1/zhangxin/Datasets/voc/', download=False)
    # imagenet_ds = ImagenetSegDataset()

    loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=1,  # must be 1 when ViT is used since Chefer's ViT codes doesn't support batch interpretation
        shuffle=False)

    # model = resnet(pretrained=True).cuda()
    # model = vgg16(pretrained=True).cuda()
    # model = vit_lrp(pretrained=True).cuda()
    model = vit_lrp_exp(pretrained=True).cuda()
    # model = vgg19(pretrained=True).cuda()
    model.eval()

    eval(model, loader)
