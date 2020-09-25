import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, tool
from pixelssl.nn import func
from pixelssl.nn.module import patch_replication_callback

from . import ssl_base


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)



def ssl_cct(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_CCT should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_CCT, the key of element_dict should be \'model\',\n'
                'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLCCT(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLCCT(ssl_base._SSLBase):
    NAME = 'ssl_cct'
    SUPPORTED_TASK_TYPES = [CLASSIFICATION]

    def __init__(self, args):
        super(SSLCCT, self).__init__(args)

        self.main_model = None
        self.auxilary_decoders = None

        self.model = None
        self.optimizer = None
        self.lrer = None
        self.criterion = None

        self.cons_criterion = None

        # check SSL arguments

    def __build__(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create the main task model
        self.main_model = func.create_model(model_funcs[0], 'main_model', args=self.args).module
        
        # create the auxilary decoders
        ad_upscale, ad_in_channels, ad_out_channels = self.task_func.sslcct_ad_arguments() # TODO:

        vat_decoders = [
            VATDecoder(ad_upscale, ad_in_channels, ad_out_channels, xi=self.args.vat_dec_xi, eps=self.args.vat_dec_eps) \
                for _ in range(0, self.args.vat_dec_num)
        ]
        drop_decoders = [
            DropOutDecoder(ad_upscale, ad_in_channels, ad_out_channels, drop_rate=self.args.drop_dec_rate, spatial_dropout=self.args.drop_dec_spatial) \
                for _ in range(0, self.args.drop_dec_num)
        ]
        cut_decoders = [
            CutOutDecoder(ad_upscale, ad_in_channels, ad_out_channels, erase=self.args.cut_dec_erase) \
                for _ in range(0, self.cut_dec_num)
        ]
        context_decoders = [
            ContextMaskingDecoder(ad_upscale, ad_in_channels, ad_out_channels) \
                for _ in range(0, self.args.context_dec_num)
        ]
        object_decoders = [
            ObjectMaskingDecoder(ad_upscale, ad_in_channels, ad_out_channels) \
                for _ in range(0, self.args.object_dec_num)
        ]
        feature_drop_decoders = [
            FeatureDropDecoder(ad_upscale, ad_in_channels, ad_out_channels) \
                for _ in range(0, self.args.fd_dec_num)
        ]
        feature_noise_decoders = [
            FeatureNoiseDecoder(ad_upscale, ad_in_channels, ad_out_channels, uniform_range=self.args.fn_dec_uniform) \
                for _ in range(0, self.args.fdn_dec_num)
        ]

        self.auxilary_decoders = nn.ModuleList(
            [*vat_decoders, *drop_decoders, *cut_decoders, *context_decoders, *object_decoders, *feature_drop_decoders, *feature_noise_decoders]
        )

        # wrap 'self.main_model' and 'self.auxilary decoders' into a single model
        self.model = WrappedCCTModel(self.args, self.main_model, self.auxilary_decoders)
        self.model = nn.DataParallel(self.model).cuda()

        # call 'patch_replication_callback' to use the `sync_batchnorm` layer
        patch_replication_callback(self.model)
        self.models = {'model': self.model}

        # create optimizers
        self.optimizer = optimizer_funcs[0](self.model.module.param_groups)
        self.optimizers = {'optimizer': self.optimizer}

        # create lrers
        self.lrer = lrer_funcs[0](self.optimizer)
        self.lrers = {'lrer': self.lrer}

        # create criterions
        self.criterion = criterion_funcs[0](self.args)
        # TODO: support more types of the consistency criterion
        self.cons_criterion = nn.MSELoss()
        self.criterions = {'criterion': self.criterion, 'cons_criterion': self.cons_criterion}

        self._algorithm_warn()

    
    def _train(self, data_loader, epoch):
        pass

    def _validate(self, data_loader, epoch):
        pass

    def _save_checkpoint(self, epoch):
        pass

    def _load_checkpoint(self):
        pass

    def _algorithm_warn(self):
        pass


class WrappedCCTModel(nn.Module):
    def __init__(self, args, main_model, auxilary_decoders):
        super(WrappedCCTModel, self).__init__()
        self.args = args
        self.main_model = main_model
        self.auxilary_decoders = auxilary_decoders

        self.param_groups = self.main_model.param_groups + \
            [{'params': self.auxilary_decoders.parameters(), 'lr': self.args.lr}]   # TODO: arguments, scale lr!!!

    def forward(self, inp):
        resulter, debugger = {}, {}

        m_resulter, m_debugger = self.main_model.forward(inp)

        if not 'pred' in t_resulter.keys() or not 'activated_pred' in t_resulter.keys():
            logger.log_err('In SSL_CCT, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                           '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                           '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                           'We need both of them since some losses include the activation functions,\n'
                           'e.g., the CrossEntropyLoss has contained SoftMax\n')

        resulter['pred'] = tool.dict_value(m_resulter, 'pred')
        resulter['activated_pred'] = tool.dict_value(m_resulter, 'activated_pred')

        if not 'sslcct_ad_inp' in m_resulter.keys():
            logger.log_err('In SSL_CCT, the \'resulter\' dict returned by the task model should contain the key:\n'
                           '    \'sslcct_ad_inp\'\t=>\tinputs of the auxilary_decoders (a 4-dim tensor)\n'
                           'It is the feature map encoded by the task model\n'
                           'Please add the key \'sslcct_ad_inp\' in your task model\'s resulter\n'
                           'Note that for different task models, the shape of \'sslcct_ad_inp\' may be different\n')
            
        ad_inp = tool.dict_value(m_resulter, 'sslcct_ad_inp')

        unlabeled_ad_inp = func.split_tensor_tuple(ad_inp, self.args.labeled_batch_size, self.args.batch_size)
        unlabeled_main_pred = func.split_tensor_tuple(resulter['pred'], self.args.labeled_batch_size, self.args.batch_size)

        # TODO: warn, only support the task model with one pred!!!
        unlabeled_ad_preds = []
        for ad in self.auxilary_decoders:
            unlabeled_ad_preds.append(ad.forward(unlabeled_ad_inp, pred_of_main_decoder=unlabeled_main_pred[0].detach()))

        resulter['unlabeled_ad_preds'] = unlabeled_ad_preds

        return resulter, debugger


# =======================================================
# Archtectures of the Auxilary Decoders
#   Following code is adapted form the repository:
#       https://github.com/yassouali/CCT 
# =======================================================

# TODO: upscale and in_channels are the arguments of auxilary encoders, write in the task_func


class VATDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        r_adv = self.get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        x = self.upsample(x + r_adv)
        return x

    def get_r_adv(self, x, decoder, it=1, xi=1e-1, eps=10.0):
        """ Virtual Adversarial Training from:
                https://arxiv.org/abs/1704.03976
        """
        x_detached = x.detach()
        with torch.no_grad():
            pred = F.softmax(decoder(x_detached), dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        for _ in range(it):
            d.requires_grad_()
            pred_hat = decoder(x_detached + xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            decoder.zero_grad()

        r_adv = d * eps
        return r_adv


class DropOutDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x = self.upsample(self.dropout(x))
        return x


class CutOutDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        maskcut = self.guided_cutout(pred_of_main_decoder, upscale=self.upscale, 
                                     erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = self.upsample(x)
        return x

    def guided_cutout(self, output, upscale, resize, erase=0.4, use_dropout=False):
        if len(output.shape) == 3:
            masks = (output > 0).float()
        else:
            masks = (output.argmax(1) > 0).float()

        if use_dropout:
            p_drop = random.randint(3, 6)/10
            maskdroped = (F.dropout(masks, p_drop) > 0).float()
            maskdroped = maskdroped + (1 - masks)
            maskdroped.unsqueeze_(0)
            maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

        masks_np = []
        for mask in masks:
            mask_np = np.uint8(mask.cpu().numpy())
            mask_ones = np.ones_like(mask_np)
            try: # Version 3.x
                _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except: # Version 4.x
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
            for poly in polys:
                min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
                min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
                bb_w, bb_h = max_w-min_w, max_h-min_h
                rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
                rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
                h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
                w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
                mask_ones[h_start:h_end, w_start:w_end] = 0
            masks_np.append(mask_ones)
        masks_np = np.stack(masks_np)

        maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
        maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

        if use_dropout:
            return maskcut.to(output.device), maskdroped.to(output.device)
        return maskcut.to(output.device)


class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x_masked_context = self.guided_masking(x, pred_of_main_decoder, resize=(x.size(2), x.size(3)),
                                               upscale=self.upscale, return_msk_context=True)
        x_masked_context = self.upsample(x_masked_context)
        return x_masked_context

    def guided_masking(self, x, output, upscale, resize, return_msk_context=True):
        if len(output.shape) == 3:
            masks_context = (output > 0).float().unsqueeze(1)
        else:
            masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
        
        masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

        x_masked_context = masks_context * x
        if return_msk_context:
            return x_masked_context

        masks_objects = (1 - masks_context)
        x_masked_objects = masks_objects * x
        return x_masked_objects


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x_masked_obj = self.guided_masking(x, pred_of_main_decoder, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
        x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj

    def guided_masking(self, x, output, upscale, resize, return_msk_context=True):
        if len(output.shape) == 3:
            masks_context = (output > 0).float().unsqueeze(1)
        else:
            masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
        
        masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

        x_masked_context = masks_context * x
        if return_msk_context:
            return x_masked_context

        masks_objects = (1 - masks_context)
        x_masked_objects = masks_objects * x
        return x_masked_objects


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x = self.feature_dropout(x)
        x = self.upsample(x)
        return x

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def forward(self, x, pred_of_main_decoder=None):
        x = self.feature_based_noise(x)
        x = self.upsample(x)
        return x

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise
