import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class RegLoss_position(nn.Module):
    def __init__(self, lambda_xy = 1.):
        super(RegLoss_position,self).__init__()
        self.lambda_xy = lambda_xy

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) [num_boxes, 2]
        target_tensor: (tensor) [num_boxes, 2]
        '''
        pred_tensor = torch.abs(pred_tensor)
        pred_tensor = pred_tensor[1::2]
        new_target_tensor = target_tensor[1::2]
        non_ignore_mask = new_target_tensor[:, 0] != 2.
        target_tensor_nig = new_target_tensor[non_ignore_mask].reshape(-1, 2)
        pred_tensor_nig = pred_tensor[non_ignore_mask].reshape(-1, 2)
        assert len(target_tensor_nig) == len(pred_tensor_nig)
        xy_loss = F.mse_loss(pred_tensor_nig[:,:2],
                                target_tensor_nig[:,:2],
                                reduction='sum')

        return self.lambda_xy*xy_loss

class RegLoss_size(nn.Module):
    def __init__(self, lambda_wh = 1.):
        super(RegLoss_size,self).__init__()
        self.lambda_wh = lambda_wh

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) [num_boxes, 2]
        target_tensor: (tensor) [num_boxes, 2]
        '''
        pred_tensor = torch.abs(pred_tensor)
        pred_tensor = pred_tensor[1::2]
        new_target_tensor = target_tensor[1::2]
        non_ignore_mask = new_target_tensor[:, 0] != 2.
        target_tensor_nig = new_target_tensor[non_ignore_mask].reshape(-1, 2)
        pred_tensor_nig = pred_tensor[non_ignore_mask].reshape(-1, 2)
        assert len(target_tensor_nig) == len(pred_tensor_nig)
        wh_loss = F.mse_loss(pred_tensor_nig[:,:2],
                                target_tensor_nig[:,:2],
                                reduction='sum')

        return self.lambda_wh*wh_loss

class RegLoss(nn.Module):
    def __init__(self, reduction = 'sum', pretrain=False, lambda_xy = 1.,
                lambda_wh = 1., refine = False):
        super(RegLoss,self).__init__()
        self.reduction = reduction
        self.pretrain = pretrain
        self.lambda_xy = lambda_xy
        self.lambda_wh = lambda_wh
        self.refine = refine
        self.clip = False

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) [num_boxes, 4]
        target_tensor: (tensor) [num_boxes, 4]
        '''
        pred_tensor = torch.abs(pred_tensor)
        if self.pretrain:
            pred_tensor = pred_tensor[1::2]
            new_target_tensor = target_tensor[1::2]
            if self.clip:
                pred_tensor[:, [0,2]] = torch.div(torch.round(torch.mul(pred_tensor[:, [0,2]],288)),288)
                pred_tensor[:, [1,3]] = torch.div(torch.round(torch.mul(pred_tensor[:, [1,3]],512)),512)
            non_ignore_mask = new_target_tensor[:, 0] != 2.
#                 assert len(non_ignore_mask) % 2 == 0, "pretrain boxes should be paired!"
#                 pred_tensor_nig = pred_tensor[non_ignore_mask].reshape(-1, 4)
            target_tensor_nig = new_target_tensor[non_ignore_mask].reshape(-1, 4)
            pred_tensor_nig = pred_tensor[non_ignore_mask].reshape(-1, 4)
            # print("Target_xy", target_tensor_nig[0, :2] + 0.5*target_tensor_nig[0,2:])
            # print("Pred_xy", pred_tensor_nig[0, :2])
            # print("Target_wh", target_tensor_nig[0, 2:])
            # print("Pred_wh", pred_tensor_nig[0, 2:])
            num_boxes_nig = pred_tensor_nig.shape[0]
            target_wh = target_tensor_nig[:,2:].clone()
            target_wh[:,1] *= target_wh[:,0]
            assert len(target_tensor_nig) == len(pred_tensor_nig)
            xy_loss = F.mse_loss(pred_tensor_nig[:,:2],
                                 target_tensor_nig[:,:2],
                                 reduction='sum') 
            wh_loss = F.mse_loss(torch.sqrt(pred_tensor_nig[:,2:4]),
                            torch.sqrt(target_tensor_nig[:,2:4]),reduction='sum') 
            if torch.isnan(xy_loss).any():
                print("XY", xy_loss, "WH", wh_loss)
                print("pred size", pred_tensor.size())
                print('target size', target_tensor.size())
                print("pred array", pred_tensor_nig)
                print("target array", target_tensor_nig)
        else:
            num_boxes = pred_tensor.shape[0] # batch size
            non_ignore_mask = torch.ones(num_boxes)
            
            if len((target_tensor == 2.).nonzero()) != 0:
                mask_start_index = (target_tensor == 2.).nonzero()[0][0]
                non_ignore_mask[mask_start_index:] = 0
                non_ignore_mask = non_ignore_mask.bool()

                pred_tensor_nig = pred_tensor[non_ignore_mask].reshape(-1, 4)
                target_tensor_nig = target_tensor[non_ignore_mask].reshape(-1, 4)
            else:
                pred_tensor_nig = pred_tensor
                target_tensor_nig = target_tensor
                
                
            num_boxes_nig = pred_tensor_nig.shape[0]

            xy_loss = F.mse_loss(pred_tensor_nig[:,:2],
                                 target_tensor_nig[:,:2],
                                 reduction='sum')
            wh_loss = F.mse_loss(torch.sqrt(pred_tensor_nig[:,2:4]),
                                     torch.sqrt(target_tensor_nig[:,2:4]),reduction='sum') 

        if self.reduction == 'mean':
            return (self.lambda_xy * xy_loss + self.lambda_wh * wh_loss) / num_boxes_nig, 0
        elif self.reduction == 'sum':
            #print(self.lambda_xy, xy_loss, self.lambda_wh, wh_loss)
            return self.lambda_xy * xy_loss + self.lambda_wh * wh_loss, 0
        else:
            assert False, 'We do not support {} reduction!'.format(self.reduction)

class Log_Pdf(nn.Module):
    def __init__(self, reduction = 'sum', pretrain=False, lambda_xy = 1.,
                lambda_wh = 1., rel_gt = False, raw_batch_size=64, KD_ON=False,
                Topk=-1):
        super(Log_Pdf,self).__init__()
        self.reduction = reduction
        self.pretrain = pretrain
        self.lambda_xy = lambda_xy
        self.lambda_wh = lambda_wh
        self.rel_gt = rel_gt
        self.gmm_comp_num = 5
        self.raw_batch_size=raw_batch_size
        self.grid_sample = False
        self.grid_size = (8, 8)
        self.KD_ON = KD_ON
        self.Topk = Topk
        self.mu2 = None
        self.sigma_2 = None
        self.kl_loss = None
        self.topk_mask = None
        self.sigma_diag_1 = None
        self.sigma_diag_2 = None
        self.raw_pdf = None
        self.xv = None
        self.yv = None
    
    def forward(self, input_gmm, input_xywh, only_wh, only_xy):
        if not self.rel_gt:
            gmm = input_gmm[1::2]
            xywh = input_xywh[1::2]

#         gmm = n_gmm[:,2:]
        non_ignore_mask = xywh[:, 0] != 2.
        assert len(non_ignore_mask) % 2 == 0, "pretrain boxes should be paired!"
        gmm = gmm[non_ignore_mask]
        xywh = xywh[non_ignore_mask]

        xy_gmm = gmm[:, :self.gmm_comp_num*6]
        wh_gmm = gmm[:, self.gmm_comp_num*6:]
        
        if only_wh: 
            gt_x = xywh[:, 2]
            gt_y = xywh[:, 3]
            gt_w = xywh[:, 2]
            gt_h = xywh[:, 3]
        elif only_xy:
            gt_x = xywh[:, 0]
            gt_y = xywh[:, 1]
            gt_w = xywh[:, 0]
            gt_h = xywh[:, 1]
        else:
            gt_x = xywh[:, 0]
            gt_y = xywh[:, 1]
            gt_w = xywh[:, 2]
            gt_h = xywh[:, 3]
            
        
        
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = self.get_gmm_params(xy_gmm)
        pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = self.get_gmm_params(wh_gmm)

        batch_size, gmm_comp_num = pi_xy.size()
            
        if self.grid_sample:
            # 3. calculate the bbox loss
            total_grid_num = self.grid_size[0]*self.grid_size[1]
            gt_x = gt_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, total_grid_num)
            gt_y = gt_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, total_grid_num)
            gt_w = gt_w.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, total_grid_num)
            gt_h = gt_h.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, total_grid_num)
            
            xy_pdf = self.grid_sample_pdf(pi_xy, gt_x, gt_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num, self.grid_size)
            wh_pdf = self.grid_sample_pdf(pi_wh, gt_w, gt_h, u_w, u_h, sigma_w, sigma_h, rho_wh, batch_size, gmm_comp_num, self.grid_size)
            
        else:
            # 3. calculate the bbox loss
            gt_x = gt_x.unsqueeze(1).repeat(1, gmm_comp_num)
            gt_y = gt_y.unsqueeze(1).repeat(1, gmm_comp_num)
            gt_w = gt_w.unsqueeze(1).repeat(1, gmm_comp_num)
            gt_h = gt_h.unsqueeze(1).repeat(1, gmm_comp_num)
        
            xy_pdf = self.pdf(pi_xy, gt_x, gt_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num)
            wh_pdf = self.pdf(pi_wh, gt_w, gt_h, u_w, u_h, sigma_w, sigma_h, rho_wh, batch_size, gmm_comp_num)
        bbox_loss = -(torch.sum(xy_pdf)*self.lambda_xy)-(torch.sum(wh_pdf)*self.lambda_wh)
        kl_loss = torch.tensor(0.)
        if self.KD_ON:
            mu = torch.cat((u_x.unsqueeze(-1),u_y.unsqueeze(-1)),-1)
            sigma = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

            kl_loss = self.batch_Bivar_KLDivLoss(pi_xy, mu, sigma)
            mu = torch.cat((u_w.unsqueeze(-1),u_h.unsqueeze(-1)),-1)
            sigma = torch.cat((sigma_w.unsqueeze(-1),sigma_h.unsqueeze(-1)),-1)
            kl_loss += self.batch_Bivar_KLDivLoss(pi_wh, mu, sigma)

        if self.reduction == 'mean':
            loss = {'bbox_loss':(bbox_loss/ batch_size) * self.raw_batch_size, 
                    'kl_loss': (kl_loss/ batch_size) * self.raw_batch_size}
            return loss['bbox_loss'], loss['kl_loss']
        elif self.reduction == 'sum':
            return bbox_loss, kl_loss
        
    def batch_Bivar_KLDivLoss(self, pi, mu1, sigma_1):
        """
        mu1 == mu2 == [batch * sentence_length, gmm_comp_num, 2]
        sigma_1 == sigma_2 == [batch * sentence_length, gmm_comp_num, 2]
        """
        # [t.inverse() for t in torch.functional.split(a,2)
        # torch.stack(b)
        self.mu2 = mu1.clone().to(mu1.device)
        self.sigma_2 = torch.ones(sigma_1.size()).to(sigma_1.device) * 1.
        total_num = mu1.size(0)
        self.kl_loss = torch.Tensor([0]).cuda().squeeze()
        for i in range(mu1.shape[1]):
#             sig_pi = pi[:,i]
            sig_mu1, sig_mu2 = mu1[:,i,:], self.mu2[:,i,:]
            sig_sigma_1, sig_sigma_2 = sigma_1[:,i,:], self.sigma_2[:,i,:]
            self.sigma_diag_1 = torch.eye(sig_sigma_1.shape[1]).unsqueeze(0).repeat(total_num,1,1).cuda() * sig_sigma_1.unsqueeze(-1)
            self.sigma_diag_2 = torch.eye(sig_sigma_2.shape[1]).unsqueeze(0).repeat(total_num,1,1).cuda() * sig_sigma_2.unsqueeze(-1)
            # sigma_diag_2_inv = [total_num, 2, 2]
            sigma_diag_2_inv = self.sigma_diag_2.inverse()
            term0 = torch.log(torch.diagonal(self.sigma_diag_2,dim1=-1,dim2=-2).prod(1) / torch.diagonal(self.sigma_diag_1,dim1=-1,dim2=-2).prod(1))
            term1 = torch.ones(sig_mu1.size(0)).to(sig_mu1.device) * sig_mu1.size(-1)
            term2 = torch.einsum('bii->b', torch.bmm(sigma_diag_2_inv, self.sigma_diag_1))
            term3_0 = torch.bmm((sig_mu2 - sig_mu1).unsqueeze(1), sigma_diag_2_inv)
            term3 = torch.bmm(term3_0, (sig_mu2 - sig_mu1).unsqueeze(-1)).view(-1)
            self.kl_loss +=  (0.5 * (term0 - term1 + term2 + term3)).sum()
        return self.kl_loss


    def get_gmm_params(self, gmm_params):
        '''
        Args:
            gmm_params: B x gmm_comp_num*gmm_param_num (B, 5*6)
        '''
        # Each: (B, 5)
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(gmm_params, self.gmm_comp_num, dim=1)
#         u_x = nn.Sigmoid()(u_x)
#         u_y = nn.Sigmoid()(u_y)
#         sigma_x = sigma_x.clamp(max=0)
#         sigma_y = sigma_y.clamp(max=0)
        pi = nn.Softmax(dim=1)(pi)
#         pi = nn.Sigmoid()(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho_xy = torch.tanh(rho_xy)

        return (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)

    def old_pdf(self, pi_xy, x, y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num):
        '''
        pdf code in Obj-GAN
        '''
        # all inputs have the same shape: batch*gmm_comp_num
        z_x = ((x-u_x)/sigma_x)**2
        z_y = ((y-u_y)/sigma_y)**2
        z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
        z = z_x + z_y - 2*rho_xy*z_xy
        a = -z/(2*(1-rho_xy**2))
        a = a.view(batch_size, gmm_comp_num)
        a_max = torch.max(a, dim=1)[0]
        a_max = a_max.unsqueeze(1).repeat(1, gmm_comp_num)
        a, a_max = a.view(-1), a_max.view(-1)

        exp = torch.exp(a-a_max)
        norm = torch.clamp(2*math.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-5).view(-1)

        raw_pdf = pi_xy.view(-1)*exp/norm
        raw_pdf = raw_pdf.view(batch_size, gmm_comp_num)
        raw_pdf = torch.log(torch.sum(raw_pdf, dim=1)+1e-5)
        a_max = a_max.view(batch_size, gmm_comp_num)[:,0]
        raw_pdf = raw_pdf + a_max

        return raw_pdf

    def pdf(self, pi_xy, x, y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num):
        '''
        Log loss proposed in sketch-RNN and Obj-GAN
        '''
        # all inputs have the same shape: (batch, gmm_comp_num)
        if self.Topk != -1:
            k = self.Topk
            distance = ((x-u_x)**2.+(y-u_y)**2.)**0.5
            topk_indice = torch.topk(distance, k, dim=-1, largest=False)[1]
            self.topk_mask = torch.zeros(u_x.size()).to(u_x.device)
            self.topk_mask = self.topk_mask.scatter_(1,topk_indice, 1).bool()
        
        z_x = ((x-u_x)/sigma_x)**2
        z_y = ((y-u_y)/sigma_y)**2
        z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
        z = z_x + z_y - 2*rho_xy*z_xy
        a = -z/(2*(1-rho_xy**2))
        exp = torch.exp(a)

        # avoid 0 in denominator
        norm = torch.clamp(2*math.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-5)
        raw_pdf = pi_xy*exp/norm

        if self.Topk != -1:
#             raw_pdf = raw_pdf.reshape(batch_size, k)
            self.raw_pdf = raw_pdf.cpu()[self.topk_mask.cpu()].reshape(batch_size, k).cuda()
        # avoid log(0)
        raw_pdf = torch.log(torch.sum(raw_pdf, dim=1)+1e-5)

        return raw_pdf
    
    def grid_sample_pdf(self, pi_xy, gt_x, gt_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num, grid_size=(32, 32)):
        # inputs have the same shape: (batch, gmm_comp_num)
        # gt_x, gt_y = [batch, 1, total_grid_num]
        batch, comp_num = u_x.shape
        total_num_grid = grid_size[0]*grid_size[1]
        # calculate the grid center
        grid_center = []
        self.xv = torch.arange(start=0.,end=1.,step=1./grid_size[0]).to(gt_x.device)+(1./grid_size[0])/2
        self.yv = torch.arange(start=0.,end=1.,step=1./grid_size[1]).to(gt_x.device)+(1./grid_size[1])/2
        xv, yv = torch.meshgrid(xv,yv)
        
        x = xv.contiguous().view(-1).unsqueeze(0).unsqueeze(0).repeat(batch, comp_num, 1)
        y = yv.contiguous().view(-1).unsqueeze(0).unsqueeze(0).repeat(batch, comp_num, 1)
        # calculate the pdf for mesh
        pi_xy = pi_xy.unsqueeze(-1).repeat(1, 1, total_num_grid)
        u_x = u_x.unsqueeze(-1).repeat(1, 1, total_num_grid)
        u_y = u_y.unsqueeze(-1).repeat(1, 1, total_num_grid)
        sigma_x = sigma_x.unsqueeze(-1).repeat(1, 1, total_num_grid)
        sigma_y = sigma_y.unsqueeze(-1).repeat(1, 1, total_num_grid)
        rho_xy = rho_xy.unsqueeze(-1).repeat(1, 1, total_num_grid)
        # all inputs have the same shape: (batch, gmm_comp_num, total_num_grid)
        z_x = ((x-u_x)/sigma_x)**2
        z_y = ((y-u_y)/sigma_y)**2
        z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
        z = z_x + z_y - 2*rho_xy*z_xy
        a = -z/(2*(1-rho_xy**2))
        exp = torch.exp(a)
        # avoid 0 in denominator
        norm = torch.clamp(2*math.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-6)
        raw_pdf = pi_xy*exp/norm
        # avoid log(0), raw_pdf = [batch, grid_size[0]*grid_size[1]]
        raw_pdf = torch.sum(raw_pdf, dim=1)
#         print(raw_pdf[0])
        raw_prob = nn.Softmax(1)(raw_pdf*5)/5**2.

        # raw_prob = (batch, total_num_grid)
        # find nearest grid center for each gt_x, gt_y
        dx = (gt_x-xv.contiguous().view(-1).unsqueeze(0).unsqueeze(0).repeat(batch,1,1))**2.
        dy = (gt_y-yv.contiguous().view(-1).unsqueeze(0).unsqueeze(0).repeat(batch,1,1))**2.
        dgt_v = torch.sqrt(dx + dy)
        closest_v_idx = dgt_v.squeeze(1).argmin(1)
        log_prob = torch.log(raw_prob[torch.arange(raw_prob.size(0)),closest_v_idx]+1e-5) 
        del x, y, xv, yv, dgt_v, dx, dy, closest_v_idx, raw_pdf
        return log_prob
    
class Rel_Loss(nn.Module):
    def __init__(self, reduction = 'sum', raw_batch_size=64):
        super(Rel_Loss,self).__init__()
        self.reduction = reduction
        self.gmm_comp_num = 5
        self.raw_batch_size=raw_batch_size
    
    def forward(self, gmm, xywh_gt):

        gmm_box = gmm[1::2]
        gmm_rel = gmm[2::2]
        xywh_box = xywh_gt[1::2]
        xywh_rel = xywh_gt[2::2]

        non_ignore_mask = xywh_box[:, 0] != 2.
        non_ignore_rel_mask = xywh_rel[:, 0] != 2.
        
        assert non_ignore_mask.sum() % 2 == 0, "pretrain boxes should be paired!"
        
        gmm_box = gmm_box[non_ignore_mask]
        xywh_box = xywh_box[non_ignore_mask]
        gmm_rel = gmm_rel[non_ignore_rel_mask]
        xywh_rel = xywh_rel[non_ignore_rel_mask]

        xy_box_gmm = gmm_box[:, :self.gmm_comp_num*6]
        xy_rel_gmm = gmm_rel[:, :self.gmm_comp_num*6]
        
        # loss for rel gmm predictor
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = self.get_gmm_params(xy_rel_gmm)
        batch_size, gmm_comp_num = pi_xy.size()
        gt_x = xywh_rel[:, 0]
        gt_y = xywh_rel[:, 1]
        gt_x = gt_x.unsqueeze(1).repeat(1, gmm_comp_num)
        gt_y = gt_y.unsqueeze(1).repeat(1, gmm_comp_num)
        sample_rel = self.sample_box(pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy, 
                                    temp=1, greedy=True).reshape(batch_size, 2)
        xy_pdf = self.pdf(pi_xy, gt_x, gt_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num)
        rel_loss = -(torch.sum(xy_pdf))
        
        # loss for box relationship consistency
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = self.get_gmm_params(xy_box_gmm)
        batch_size, gmm_comp_num = pi_xy.size()
        sample_xy = self.sample_box(pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy, 
                                    temp=1, greedy=True).reshape(batch_size, 2)
        sub_pred = sample_xy[0::2]
        obj_pred = sample_xy[1::2]

#         rel_gt = xywh_box[0::2,:2] - xywh_box[1::2,:2]
        xy_loss = F.mse_loss(sub_pred-obj_pred, sample_rel, reduction='sum')
        
        if self.reduction == 'mean':
            return xy_loss/batch_size*self.raw_batch_size,rel_loss/batch_size* self.raw_batch_size
        elif self.reduction == 'sum':
            return xy_loss, rel_loss
        
    def sample_box(self, pi, u_x, u_y, sigma_x, sigma_y, rho_xy, temp = None, greedy=False):
        temperature = temp
        
        def adjust_temp(pi_pdf):
            pi_pdf = torch.log(pi_pdf)/temperature
            pi_pdf -= torch.max(pi_pdf)
            pi_pdf = torch.exp(pi_pdf)
            pi_pdf /= torch.sum(pi_pdf)
            return pi_pdf

        # get mixture indice:
        if temp is not None:
            pi = adjust_temp(pi)
        try:
#             pi_idx = pi.argmax(1).unsqueeze(-1)
            pi_idx = torch.multinomial(pi, 1)
        except:
            pi_idx = pi.argmax(1).unsqueeze(-1)
        # get mixture params:
        u_x = torch.gather(u_x, dim=1, index=pi_idx)
        u_y = torch.gather(u_y, dim=1, index=pi_idx)
#         if temp is not None:
#             sigma_x= torch.gather(sigma_x*temperature, dim=1, index=pi_idx)
#             sigma_y = torch.gather(sigma_y*temperature, dim=1, index=pi_idx)
#         else:
        sigma_x= torch.gather(sigma_x, dim=1, index=pi_idx)
        sigma_y = torch.gather(sigma_y, dim=1, index=pi_idx)
        rho_xy = torch.gather(rho_xy, dim=1, index=pi_idx)
        xy = self.sample_bivariate_normal(u_x, u_y, sigma_x, sigma_y, rho_xy, 
            temperature, greedy=greedy)
        return xy

    def pdf(self, pi_xy, x, y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num):
        '''
        Log loss proposed in sketch-RNN and Obj-GAN
        '''
        z_x = ((x-u_x)/sigma_x)**2
        z_y = ((y-u_y)/sigma_y)**2
        z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
        z = z_x + z_y - 2*rho_xy*z_xy
        a = -z/(2*(1-rho_xy**2))
        exp = torch.exp(a)

        # avoid 0 in denominator
        norm = torch.clamp(2*math.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-5)
        raw_pdf = pi_xy*exp/norm

        # avoid log(0)
        raw_pdf = torch.log(torch.sum(raw_pdf, dim=1)+1e-5)

        return raw_pdf
    
    def get_gmm_params(self, gmm_params):
        '''
        Args:
            gmm_params: B x gmm_comp_num*gmm_param_num (B, 5*6)
        '''
        # Each: (B, 5)
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(gmm_params, self.gmm_comp_num, dim=1)
#         u_x = nn.Sigmoid()(u_x)
#         u_y = nn.Sigmoid()(u_y)
#         sigma_x = sigma_x.clamp(max=0)
#         sigma_y = sigma_y.clamp(max=0)
        pi = nn.Softmax(dim=1)(pi)
#         pi = nn.Sigmoid()(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho_xy = torch.tanh(rho_xy)

        return (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)
    
    def sample_bivariate_normal(self, u_x, u_y, sigma_x, sigma_y, rho_xy, 
        temperature, greedy=False):
        # inputs must be floats
        if greedy:
            xy = torch.cat((u_x, u_y), dim=-1).cuda()
            return xy
        mean = torch.cat((u_x, u_y), dim=1)
        sigma_x *= math.sqrt(temperature)
        sigma_y *= math.sqrt(temperature)
        cov = torch.zeros((u_x.size(0), 2, 2)).cuda().detach().cpu()
        cov[:, 0, 0] = sigma_x.flatten() * sigma_x.flatten()
        cov[:, 0, 1] = rho_xy.flatten() * sigma_x.flatten() * sigma_y.flatten()
        cov[:, 1, 0] = rho_xy.flatten() * sigma_x.flatten() * sigma_y.flatten()
        cov[:, 1, 1] = sigma_y.flatten() * sigma_y.flatten()
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        singular_idx = (det == 0).nonzero()
        for idx in singular_idx:
            cov[idx] *= 0.
            cov[idx, 0, 0] += 1.
            cov[idx, 1, 1] += 1.
        m = MultivariateNormal(loc=mean, covariance_matrix=cov)
        x = m.sample()
        return x.cuda()


            
class FocalLoss(nn.Module):
    """
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, 
        respectively.
    input : [num_vocab, classes]
    target : [num_vocab,]
    """
    def __init__(self, gamma=2, alpha=None, reduction='sum', ignore_index = None):
        super(FocalLoss, self).__init__()
        self.focal_loss_alpha = alpha
        self.focal_loss_gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs_all, targets_all): 
        if self.ignore_index is not None:
            ignore_mask = targets_all != self.ignore_index
            targets = targets_all[ignore_mask]
            inputs = inputs_all[ignore_mask]
        else:
            inputs = inputs_all
            targets = targets_all
        targets_one_hot = torch.zeros(inputs.size())
        for i in range(len(targets)):
            targets_one_hot[i,targets[i]] = 1.
#         targets = targets_one_hot.cuda()
        gpu_targets = targets_one_hot.cuda()

        focal_weight = torch.where(torch.eq(gpu_targets, 1), 1. - inputs, inputs)
        if self.focal_loss_alpha is not None:
            alpha_factor = torch.ones(gpu_targets.shape) * self.focal_loss_alpha
            alpha_factor = torch.where(torch.eq(gpu_targets, 1), alpha_factor, 1. - alpha_factor)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.focal_loss_gamma)
        else:
            focal_weight = torch.pow(focal_weight, self.focal_loss_gamma)

        bce = F.binary_cross_entropy(nn.Softmax(1)(inputs), gpu_targets)
        focal_weight = focal_weight
        cls_loss = focal_weight * bce
        if self.reduction == 'mean':
            return cls_loss.mean()
        else:
            return cls_loss.sum()
        
        
class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.1):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".
        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=log_probs.size(-1))
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                == targets.shape
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss


class OverlapLoss(nn.Module):

    def __init__(self, reduction='sum', batch_size=32, sentence_size=128):
        super(OverlapLoss, self).__init__()
        self.reduction = reduction
        self.batch_size = batch_size

    def forward(self, pred_boxes, id, parent_id, type_id):
        
        sentence_size = id.size(1)
        pred_boxes = pred_boxes.reshape(-1,4)
        pred_boxes = pred_boxes[1::2]
        type_id = type_id[1::2]
        order = torch.mul(torch.arange(0, id.size(0), device='cuda:0'), sentence_size/2)
        repeated_tensor = order.repeat_interleave(id.size(1))
        reshaped_tensor = repeated_tensor.reshape(-1, id.size(1))
        id = torch.add(id, reshaped_tensor)
        parent_id = torch.add(parent_id, reshaped_tensor)
        
        id = id.reshape(id.size(0) * id.size(1))
        id = id[1::2].long()
        parent_id = parent_id.reshape(parent_id.size(0) * parent_id.size(1))
        parent_id = parent_id[1::2].long()
        #non_ignore_mask = type_id[:] != 3
        pred_boxes_xyxy = self.xcycwh2xyxy(pred_boxes,image_wh=[1440,2560])
        target_boxes_xyxy = torch.clone(pred_boxes_xyxy)
        unique_types = torch.unique(id)
        test = torch.clone(unique_types)
        for i in range(unique_types.shape[0]):
            test[i] = torch.where(id == unique_types[i])[0][-1]
        unique_types_parent = parent_id[test.long()]
        unique_id = torch.clone(unique_types)
        unique_types_p = torch.unique(torch.where(id == unique_types.view(-1,1))[0])
        for i in range(unique_types_p.shape[0]):
            unique_id[i] = torch.where(id == unique_types.view(-1,1))[1][torch.where(torch.where(id == unique_types.view(-1,1))[0] == unique_types_p[i])[0][-1]]
        
        pred_boxes_xyxy = pred_boxes_xyxy[unique_id.long()]
        unique_id = torch.clone(unique_types_parent)
        unique_types_p = torch.unique(torch.where(id == unique_types_parent.view(-1,1))[0])
        for i in range(unique_types_p.shape[0]):
            unique_id[i] = torch.where(id == unique_types_parent.view(-1,1))[1][torch.where(torch.where(id == unique_types_parent.view(-1,1))[0] == unique_types_p[i])[0][-1]]
        
        target_boxes_xyxy = target_boxes_xyxy[unique_id.long()]
        
        total_iou = self.get_iou(pred_boxes_xyxy[unique_types%(sentence_size/2)!=0,:], target_boxes_xyxy[unique_types%(sentence_size/2)!=0,:])
        if self.reduction == 'sum':
            return len(target_boxes_xyxy)-total_iou
        else:
            return 1-total_iou /  len(target_boxes_xyxy)

    def xcycwh2xyxy(self, boxes, image_wh):
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        center = boxes[:,:2].clone()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
#         boxes[:,0] *= image_wh[0]
#         boxes[:,1] *= image_wh[1]
#         boxes[:,2] *= image_wh[0]
#         boxes[:,3] *= image_wh[0]
        return boxes


    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : B * 4
            Keys: ['x1', 'x2', 'y1', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : B * 4
            Keys: ['x1', 'x2', 'y1', 'y2']
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """

        # determine the coordinates of the intersection rectangle
        x_left = torch.max(bb1[:,0], bb2[:,0])
        y_top = torch.max(bb1[:,1], bb2[:,1])
        x_right = torch.min(bb1[:,2], bb2[:,2])
        y_bottom = torch.min(bb1[:,3], bb2[:,3])

        legal_index = (x_right >= x_left) & (y_bottom >= y_top)
        x_left = x_left[legal_index]
        y_top = y_top[legal_index]
        x_right = x_right[legal_index]
        y_bottom = y_bottom[legal_index]
        bb1 = bb1[legal_index, :]
        bb2 = bb2[legal_index, :]
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1])
        bb2_area = (bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / bb1_area
        legal_num = iou.size(0)
        iou = torch.sum(iou, dim=0)
        if legal_num == 0 or not iou/legal_num >= 0.0 or not iou/legal_num <= 1.0:
            return 0

        return iou
    
    
class OverlapLoss_intra(nn.Module):

    def __init__(self, reduction='sum', batch_size=32, sentence_size=128):
        super(OverlapLoss_intra, self).__init__()
        self.reduction = reduction
        self.batch_size = batch_size

    def forward(self, pred_boxes, id, parent_id, type_id):

        sentence_size = id.size(1)
        pred_boxes = pred_boxes.reshape(-1,4)
        pred_boxes = pred_boxes[1::2]
        type_id = type_id[1::2]
        order = torch.mul(torch.arange(0, id.size(0), device='cuda:0'), sentence_size/2)
        repeated_tensor = order.repeat_interleave(id.size(1))
        reshaped_tensor = repeated_tensor.reshape(-1, id.size(1))
        id = torch.add(id, reshaped_tensor)
        parent_id = torch.add(parent_id, reshaped_tensor)
        
        id = id.reshape(id.size(0) * id.size(1))
        id = id[1::2].long()
        parent_id = parent_id.reshape(parent_id.size(0) * parent_id.size(1))
        parent_id = parent_id[1::2].long()
        #non_ignore_mask = type_id[:] != 3
        pred_boxes_xyxy = self.xcycwh2xyxy(pred_boxes,image_wh=[1440,2560])
        target_boxes_xyxy = torch.clone(pred_boxes_xyxy)
        #print("old pred boxes")
        #print(pred_boxes_xyxy)

        unique_types = torch.unique(id)
        test = torch.clone(unique_types)
        for i in range(unique_types.shape[0]):
            test[i] = torch.where(id == unique_types[i])[0][-1]
        unique_types_parent = parent_id[test.long()]
        unique_parent, _ = torch.unique(unique_types_parent, return_counts=True)
        
        x1_list = []
        x2_list = []
        
        it = 0
        for i in range(unique_parent.shape[0]):
            unique_id_by_parent = unique_types[torch.where(unique_types_parent==unique_parent[i])[0]]
            k = unique_id_by_parent.size(0) - 1
            k_p = k
            while k > 0:
                x1_list.append(unique_id_by_parent[(unique_id_by_parent.size(0) - 1 - k)])
                x2_list.append(unique_id_by_parent[(unique_id_by_parent.size(0) - k_p)])
                k_p -= 1
                if k_p == 0:
                    k -= 1
                    k_p = k
                it += 1
        x1 = torch.tensor(x1_list, dtype=torch.int, device='cuda:0')
        x2 = torch.tensor(x2_list, dtype=torch.int, device='cuda:0')
        unique_id = torch.clone(x1)
        unique_types_p = torch.unique(torch.where(id == x1.view(-1,1))[0])
        for i in range(unique_types_p.shape[0]):
            unique_id[i] = torch.where(id == x1.view(-1,1))[1][torch.where(torch.where(id == x1.view(-1,1))[0] == unique_types_p[i])[0][-1]]
        pred_boxes_xyxy = pred_boxes_xyxy[unique_id.long()]
        
        unique_id = torch.clone(x2)
        unique_types_p = torch.unique(torch.where(id == x2.view(-1,1))[0])
        for i in range(unique_types_p.shape[0]):
            unique_id[i] = torch.where(id == x2.view(-1,1))[1][torch.where(torch.where(id == x2.view(-1,1))[0] == unique_types_p[i])[0][-1]]
        target_boxes_xyxy = target_boxes_xyxy[unique_id.long()]
        total_iou = self.get_iou(pred_boxes_xyxy, target_boxes_xyxy)
        if self.reduction == 'sum':
            return total_iou
        else:
            return total_iou /  len(target_boxes_xyxy)

    def xcycwh2xyxy(self, boxes, image_wh):
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        center = boxes[:,:2].clone()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
#         boxes[:,0] *= image_wh[0]
#         boxes[:,1] *= image_wh[1]
#         boxes[:,2] *= image_wh[0]
#         boxes[:,3] *= image_wh[0]
        return boxes


    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : B * 4
            Keys: ['x1', 'x2', 'y1', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : B * 4
            Keys: ['x1', 'x2', 'y1', 'y2']
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """

        # determine the coordinates of the intersection rectangle
        x_left = torch.max(bb1[:,0], bb2[:,0])
        y_top = torch.max(bb1[:,1], bb2[:,1])
        x_right = torch.min(bb1[:,2], bb2[:,2])
        y_bottom = torch.min(bb1[:,3], bb2[:,3])

        legal_index = (x_right >= x_left) & (y_bottom >= y_top)
        x_left = x_left[legal_index]
        y_top = y_top[legal_index]
        x_right = x_right[legal_index]
        y_bottom = y_bottom[legal_index]
        bb1 = bb1[legal_index, :]
        bb2 = bb2[legal_index, :]
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1])
        bb2_area = (bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / torch.minimum(bb1_area,bb2_area)
        legal_num = iou.size(0)
        iou = torch.sum(iou, dim=0)
        if legal_num == 0 or not iou/legal_num >= 0.0 or not iou/legal_num <= 1.0:
            return 0

        return iou



#pour la loss des relations
# inside: x0children-x0parent > 0 x1c-x1p < 0 y0children-y0parent >0 y1c-y1p < 0 
# right to or left to if: abs(x02 - x0) < abs(y02 - y01)
# check positive or negative to determine the direction