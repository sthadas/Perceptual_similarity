import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from util import util
from skimage import color
from IPython import embed
from . import pretrained_networks as pn
from scipy import signal


# Off-the-shelf deep network
class PNet(nn.Module):
    '''Pre-trained network with all channels equally weighted by default'''
    def __init__(self, pnet_type='vgg', pnet_rand=False, use_gpu=True,alt="Alt2"):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu
        self.alt = alt
        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))
        
        if(self.pnet_type in ['vgg','vgg16']):
            self.net = pn.vgg16(pretrained=not self.pnet_rand,requires_grad=False)
        elif(self.pnet_type=='alex'):
            self.net = pn.alexnet(pretrained=not self.pnet_rand,requires_grad=False)
        elif(self.pnet_type[:-2]=='resnet'):
            self.net = pn.resnet(pretrained=not self.pnet_rand,requires_grad=False, num=int(self.pnet_type[-2:]))
        elif(self.pnet_type=='squeeze'):
            self.net = pn.squeezenet(pretrained=not self.pnet_rand,requires_grad=False)

        self.L = self.net.N_slices

        if(use_gpu):
            self.net.cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()

    #############################################################################################
    # Alternative 1 - Create covariance matrix of 2 features, calc the determinant and accumulate
    #############################################################################################
    def Alt1(self,kk,flat0,flat1,outs0):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            # print(idx)
            vec0 = np.concatenate(flat0[0,idx,:,:])
            vec1 = np.concatenate(flat1[0,idx,:,:])
            covmat = np.cov(vec0,vec1)
            cur_score += np.linalg.det(covmat)
        return cur_score

    ############################################################################################
    # Alternative 2 - Create covariance matrix of 2 features, calculate correlation coefficient and accumulate
    ############################################################################################
    def Alt2(self,kk,flat0,flat1,outs0,img):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            vec0 = np.concatenate(flat0[img,idx,:,:])
            vec1 = np.concatenate(flat1[img,idx,:,:])
            covmat = np.cov(vec0,vec1)
            if covmat[0, 0]!=0 and covmat[1, 1] != 0:
                cur_score += np.abs(covmat[1, 0]/np.sqrt(covmat[1, 1] * covmat[0, 0])) # Absolute value of correlation coefficient
            elif covmat[0, 0]==0 and covmat[1, 1] == 0:  # Two constants -> high correlation
                cur_score += 1
        return cur_score

    ############################################################################################
    # Alternative 3 - Use the original method (cos_sim) and correlation coefficient
    ############################################################################################
    def Alt3(self,kk,flat0,flat1,outs0,outs1,img):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            vec0 = np.concatenate(flat0[img,idx,:,:])
            vec1 = np.concatenate(flat1[img,idx,:,:])
            vec0 = (vec0-np.mean(vec0))/(np.sqrt(np.cov(vec0)+1e-5))
            vec1 = (vec1-np.mean(vec1))/(np.sqrt(np.cov(vec1)+1e-5))
            covmat = np.cov(vec0,vec1)
            if covmat[0, 0] != 0 and covmat[1, 1] != 0:
                cur_score += 1. - (covmat[1, 0]/np.sqrt(covmat[1, 1] * covmat[0, 0])) # Absolute value of correlation coefficient
            elif covmat[0, 0] == 0 and covmat[1, 1] == 0:  # Two constants -> high correlation
                cur_score += 0
            else:
                cur_score += 1
        cur_score = cur_score/(((outs0[kk]).size())[1])
        cur_score_orig = (1.-util.cos_sim((outs0[kk])[img].reshape(1,(outs0[kk]).size()[1],(outs0[kk]).size()[2],(outs0[kk]).size()[3]),(outs1[kk])[img].reshape(1,(outs0[kk]).size()[1],(outs0[kk]).size()[2],(outs0[kk]).size()[3])))
        return (cur_score+cur_score_orig.item())/2

    ############################################################################################
    # Alternative 4 - Sum and compare each feature's matrix Eigenvalues
    ############################################################################################
    def Alt4(self,kk,flat0,flat1,outs0,img):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            eig_sum0 = np.linalg.eig(flat0[img, idx, :, :])[0].sum()
            eig_sum1 = np.linalg.eig(flat1[img, idx, :, :])[0].sum()
            cur_score += np.abs(eig_sum0 - eig_sum1)
        return cur_score

    ############################################################################################
    # Alternative 5 - Sum each feature's matrix values and compare
    ############################################################################################
    def Alt5(self,kk,flat0,flat1,outs0,img):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            sum0 = (flat0[img, idx, :, :]).sum()
            sum1 = (flat1[img, idx, :, :]).sum()
            cur_score += np.abs(sum0 - sum1)
        return cur_score

    ############################################################################################
    # Alternative 6 - Multiply inverse distorted matrix with ref and calc L2 with I
    ############################################################################################
    def Alt6(self, kk, flat0, flat1, outs0, img):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            psinv = np.linalg.pinv(flat1[img, idx, :, :])
            mul_res = (flat0[img, idx, :, :])*psinv
            I = np.identity(mul_res.__len__())
            cur_score += np.linalg.norm(mul_res - I)
        return cur_score

    ############################################################################################
    # Alternative 7 - Calculate correlation coefficient over rows & cols and sum up
    ############################################################################################
    def Alt7(self, kk, flat0, flat1, outs0, img):
        cur_score = 0
        for idx in range(((outs0[kk]).size())[1]):
            for row in range(((outs0[kk]).size())[2]):
                vec_rows0 = flat0[img,idx,row,:]
                vec_rows1 = flat1[img,idx,row,:]
                covmat_rows = np.cov(vec_rows0,vec_rows1)
                if covmat_rows[0, 0]!=0 and covmat_rows[1, 1] != 0:
                    cur_score += np.abs(covmat_rows[1, 0]/np.sqrt(covmat_rows[1, 1] * covmat_rows[0, 0])) # Absolute value of correlation coefficient
                elif covmat_rows[0, 0]==0 and covmat_rows[1, 1] == 0:  # Two constants -> high correlation
                    cur_score += 1
            for col in range(((outs0[kk]).size())[3]):
                vec_cols0 = flat0[img, idx, :, col]
                vec_cols1 = flat1[img, idx, :, col]
                covmat_cols = np.cov(vec_cols0, vec_cols1)
                if covmat_cols[0, 0] != 0 and covmat_cols[1, 1] != 0:
                    cur_score += np.abs(covmat_cols[1, 0] / np.sqrt(covmat_cols[1, 1] * covmat_cols[0, 0]))  # Absolute value of correlation coefficient
                elif covmat_cols[0, 0] == 0 and covmat_cols[1, 1] == 0:  # Two constants -> high correlation
                    cur_score += 1
        return cur_score

    ############################################################################################
    # Alternative 8 - Calculate correlation coefficient over a neighbourhood of 3 (optional)
    ############################################################################################
    def Alt8(self, kk, flat0, flat1, outs0, img):
        cur_score = 0
        neighbourhood = 3
        for feat in range(((outs0[kk]).size())[1]):
            for row in range(((outs0[kk]).size())[2]-(neighbourhood-1)):
                for col in range(((outs0[kk]).size())[3]-(neighbourhood-1)):
                    mini_mat0 = flat0[img,feat,row:row+neighbourhood-1,col:col+neighbourhood-1]
                    mini_mat1 = flat1[img,feat,row:row+neighbourhood-1,col:col+neighbourhood-1]
                    vec0 = np.concatenate(mini_mat0[:, :])
                    vec1 = np.concatenate(mini_mat1[:, :])
                    covmat = np.cov(vec0, vec1)
                    if covmat[0, 0] != 0 and covmat[1, 1] != 0:
                        cur_score += np.abs(covmat[1, 0] / np.sqrt(covmat[1, 1] * covmat[0, 0]))  # Absolute value of correlation coefficient
                    elif covmat[0, 0] == 0 and covmat[1, 1] == 0:  # Two constants -> high correlation
                        cur_score += 1
        return cur_score

    ############################################################################################
    #                               Tensor Ops                                                 #
    ############################################################################################

    ############################################################################################
    # Alternative 9 - Like Alt7 but with tensor operations
    ############################################################################################
    def Alt9(self, kk, outs0, outs1):
        outs0 = outs0[kk]
        outs1 = outs1[kk]
        # transposed_outs0 = torch.transpose(outs0,2,3) # transpose each feature's matrix for columns mean calculation
        # transposed_outs1 = torch.transpose(outs1,2,3) # transpose each feature's matrix for columns mean calculation
        outs0_row_means = torch.mean(outs0,3,True)
        outs0_row_means = outs0_row_means.expand_as(outs0)
        outs0_col_means = torch.mean(outs0,2,True)
        outs0_col_means = outs0_col_means.expand_as(outs0)
        # outs0_col_means = torch.mean(transposed_outs0,3)
        # outs0_col_means = torch.transpose(outs0_col_means.expand_as(outs0),2,3)
        outs1_row_means = torch.mean(outs1,3,True)
        outs1_row_means = outs1_row_means.expand_as(outs1)
        outs1_col_means = torch.mean(outs1,2,True)
        outs1_col_means = outs1_col_means.expand_as(outs1)
        # outs1_col_means = torch.mean(transposed_outs1,2,True)
        # outs1_col_means = torch.transpose(outs1_col_means.expand_as(outs1),2,3)
        outs0_row_centered = torch.add(outs0,-1,outs0_row_means)
        outs0_col_centered = torch.add(outs0,-1, outs0_col_means)
        outs1_row_centered = torch.add(outs1,-1,outs1_row_means)
        outs1_col_centered = torch.add(outs1,-1, outs1_col_means)
        # outs0_centered = torch.add(outs0,-1,outs0_row_means+outs0_col_means)
        # outs1_centered = torch.add(outs1,-1,outs1_row_means+outs1_col_means)
        # prod = torch.mul(outs0_centered,outs1_centered)
        prod_rows = torch.mul(outs0_row_centered,torch.transpose(outs1_row_centered,2,3))
        prod_cols = torch.mul(torch.transpose(outs0_col_centered,2,3),outs1_col_centered)
        cov_row_vec = torch.diagonal(prod_rows,0,2,3)
        cov_col_vec = torch.diagonal(prod_cols,0,2,3)
        cov_rows = torch.sum(cov_row_vec,2)
        cov_cols = torch.sum(cov_col_vec,2)
        outs0_row_squared = torch.pow(outs0_row_centered,2)
        outs1_row_squared = torch.pow(outs1_row_centered,2)
        outs0_col_squared = torch.pow(outs0_col_centered,2)
        outs1_col_squared = torch.pow(outs1_col_centered,2)
        outs0_row_var = torch.sqrt(torch.sum(outs0_row_squared,2))
        outs1_row_var = torch.sqrt(torch.sum(outs1_row_squared,2))
        # cov_rows = torch.sum(prod,2)
        outs0_row_var_exp = torch.unsqueeze(outs0_row_var, 2)
        outs1_row_var_exp = torch.unsqueeze(outs1_row_var, 2)
        correlation_coeff_rows = torch.div(torch.unsqueeze(torch.unsqueeze(cov_rows,2),3),torch.matmul(outs0_row_var_exp,torch.transpose(outs1_row_var_exp,2,3)))
        correlation_coeff_rows[correlation_coeff_rows != correlation_coeff_rows] = 0
        correlation_coeff_rows = torch.sum(torch.abs(correlation_coeff_rows),(2,1))
        outs0_col_var = torch.sqrt(torch.sum(outs0_col_squared,3))
        outs1_col_var = torch.sqrt(torch.sum(outs1_col_squared,3))
        outs0_col_var_exp = torch.unsqueeze(outs0_col_var, 2)
        outs1_col_var_exp = torch.unsqueeze(outs1_col_var, 2)
        # cov_cols = torch.sum(prod,3)
        correlation_coeff_cols = torch.div(torch.unsqueeze(torch.unsqueeze(cov_cols,2),3),torch.matmul(outs0_col_var_exp,torch.transpose(outs1_col_var_exp,2,3)))
        correlation_coeff_cols[correlation_coeff_cols != correlation_coeff_cols] = 0
        correlation_coeff_cols = torch.sum(torch.abs(correlation_coeff_cols),(2,1))
        # return torch.reciprocal(torch.add(correlation_coeff_cols,1,correlation_coeff_rows))
        return torch.squeeze(torch.add(correlation_coeff_cols, 1, correlation_coeff_rows))

    ############################################################################################
    # Alternative 10 - Like Alt2 but with tensor operations
    ############################################################################################
    def Alt10(self,kk, outs0, outs1):
        outs0 = outs0[kk]
        outs1 = outs1[kk]
        outs0_reshaped = torch.reshape(outs0,[(outs0.size())[0],(outs0.size())[1],(outs0.size())[2]*(outs0.size())[3]])
        outs1_reshaped = torch.reshape(outs1,[(outs1.size())[0],(outs1.size())[1],(outs1.size())[2]*(outs1.size())[3]])
        outs0_means = torch.mean(outs0_reshaped,2)
        outs0_means = torch.reshape(outs0_means,[(outs0.size())[0],(outs0.size())[1],1,1])
        outs0_means = outs0_means.expand_as(outs0)
        outs1_means = torch.mean(outs1_reshaped,2)
        outs1_means = torch.reshape(outs1_means,[(outs1.size())[0],(outs1.size())[1],1,1])
        outs1_means = outs1_means.expand_as(outs1)
        outs0_centered = torch.add(outs0,-1,outs0_means)
        outs1_centered = torch.add(outs1,-1,outs1_means)
        prod = torch.mul(outs0_centered,outs1_centered)
        outs0_squared = torch.pow(outs0_centered,2)
        outs1_squared = torch.pow(outs1_centered,2)
        outs0_mat_var = torch.sqrt(torch.sum(outs0_squared,(2,3)))
        outs1_mat_var = torch.sqrt(torch.sum(outs1_squared,(2,3)))
        cov = torch.sum(prod,(2,3))
        correlation_coeff = torch.div(cov,torch.mul(outs0_mat_var,outs1_mat_var))
        both_consts = torch.mul(correlation_coeff!=correlation_coeff,cov==0)
        correlation_coeff[correlation_coeff!=correlation_coeff] = 0
        # correlation_coeff[both_consts==1] = 1
        correlation_coeff_tot = torch.sum(torch.abs(correlation_coeff),1)
        # correlation_coeff_tot = torch.div(correlation_coeff_tot,(outs0.size())[1])
        # correlation_coeff_tot = torch.abs(torch.add(correlation_coeff_tot,-1))
        return correlation_coeff_tot

    ############################################################################################
    # Alternative 11 - Like Alt3 but with tensor operations
    ############################################################################################
    def Alt11(self,kk, outs0, outs1):
        cov_score = self.Alt10(kk,outs0,outs1)
        outs0 = outs0[kk]
        outs1 = outs1[kk]
        cov_score = torch.div(cov_score,(outs0.size())[1])
        cov_score = 1.-cov_score
        cos_sim_score = (1.-util.cos_sim(outs0,outs1))
        total_score = torch.div(torch.add(cov_score,1,cos_sim_score),2)
        return total_score

    ############################################################################################
    # Alternative 12 - Like Alt5 but with tensor operations
    ############################################################################################
    def Alt12(self,kk, outs0, outs1):
        # outs0 = outs0[kk]
        # outs1 = outs1[kk]
        outs0 = util.normalize_tensor(outs0[kk])
        outs1 = util.normalize_tensor(outs1[kk])
        outs0_sum = torch.sum(outs0,(2,3))
        outs1_sum = torch.sum(outs1,(2,3))
        diff = torch.abs(torch.add(outs0_sum,-1,outs1_sum))
        return torch.sum(diff,1)

    ############################################################################################
    # Alternative 13 - A combination of cos_sim, L1 diff, and eigVals
    ############################################################################################
    def Alt13(self,kk, outs0, outs1):
        # sum_res = self.Alt12(kk,outs0,outs1)
        # cos_sim_score = (1. - util.cos_sim(outs0, outs1))
        # cov_res = torch.reciprocal(self.Alt10(kk,outs0,outs1))
        # # return torch.stack((sum_res,cos_sim_score,cov_res),1)
        sum_res = self.Alt12(kk, outs0, outs1)
        # print("finished running alt. #1 of 3...")
        #  cos_corr = self.Alt11(kk, outs0, outs1)
        cos_sim_score = (1. - util.cos_sim(outs0[kk], outs1[kk]))
        # print("finished running alt. #2 of 3...")
        eig_vals = self.Alt18(kk,outs0[kk].cpu().data.numpy(),outs1[kk].cpu().data.numpy(),outs0)
        # eig_vals = []
        # for img in range((outs0[kk].size())[0]):
        #     eig_vals += [self.Alt17(kk,outs0[kk].cpu().data.numpy(),outs1[kk].cpu().data.numpy(),outs0,img)]
        # eig_vals = (torch.tensor(eig_vals, dtype=torch.float64,device=torch.device('cuda:0'))).float()
        # print("finished running alt. #3 of 3...")
        return torch.stack((sum_res, cos_sim_score, eig_vals), 1)

    ############################################################################################
    # Alternative 14 - Like Alt4 but with tensor operations
    ############################################################################################
    def Alt14(self,kk, outs0, outs1):
        outs0 = outs0[kk]
        outs1 = outs1[kk]
        res0 = []
        res1 = []
        for batch_size in range(outs0.size(0)):
            for feature in range(outs0.size(1)):
                eigval0, _ = torch.eig(outs0[batch_size, feature, :, :])
                res0.append(torch.max(eigval0))
                eigval1, _ = torch.eig(outs1[batch_size, feature, :, :])
                res1.append(torch.max(eigval1))
        res0 = torch.stack(res0).view(outs0.size(0), outs0.size(1))
        res1 = torch.stack(res1).view(outs1.size(0), outs1.size(1))
        return torch.sum(torch.abs(torch.add(res0,-1,res1)),1)

    ############################################################################################
    # Alternative 15 - Like Alt4 but with tensor operations and more specific calculation
    ############################################################################################
    def Alt15(self,kk, outs0, outs1):
        outs0 = outs0[kk]
        outs1 = outs1[kk]
        res = []
        for batch_size in range(outs0.size(0)):
            temp_res = []
            for feature in range(outs0.size(1)):
                eigval0, _ = torch.eig(outs0[batch_size, feature, :, :])
                sorted_eigval0, _ = torch.sort(torch.norm(eigval0,2,1),0,True);
                eigval1, _ = torch.eig(outs1[batch_size, feature, :, :])
                sorted_eigval1, _ = torch.sort(torch.norm(eigval1,2,1),0,True);
                temp_res.append(torch.dist(sorted_eigval0,sorted_eigval1).item())
            tensor_temp = torch.tensor(temp_res, dtype=torch.float64, device=torch.device('cuda:0'))
            res.append((torch.sum(tensor_temp)).item())
        return (torch.tensor(res,dtype=torch.float64,device=torch.device('cuda:0'))).float()

    ############################################################################################
    # Alternative 16 - Calculate correlation coefficient over a neighbourhood of 3 (optional) with tensor operations
    ############################################################################################
    def Alt16(self,kk, outs0, outs1):
        outs0 = outs0[kk]
        outs1 = outs1[kk]
        cur_score = 0
        neighbourhood = 3
        correlation_coeff_tot = torch.zeros((outs0.size())[0],dtype=torch.float64,device=torch.device('cuda:0')).float()
        for row in range(((outs0).size())[2]-(neighbourhood-1)):
            for col in range(((outs0).size())[3]-(neighbourhood-1)):
                mini_mat0 = outs0[:,:,row:row+neighbourhood,col:col+neighbourhood]
                mini_mat1 = outs1[:,:,row:row+neighbourhood,col:col+neighbourhood]
                mini_mat0_reshaped = torch.reshape(mini_mat0, [(mini_mat0.size())[0], (mini_mat0.size())[1],(mini_mat0.size())[2] * (mini_mat0.size())[3]])
                mini_mat1_reshaped = torch.reshape(mini_mat1, [(mini_mat1.size())[0], (mini_mat1.size())[1],(mini_mat1.size())[2] * (mini_mat1.size())[3]])
                mini_mat0_means = torch.mean(mini_mat0_reshaped, 2)
                mini_mat0_means = torch.reshape(mini_mat0_means, [(mini_mat0_means.size())[0], (mini_mat0_means.size())[1], 1, 1])
                mini_mat0_means = mini_mat0_means.expand_as(mini_mat0)
                mini_mat1_means = torch.mean(mini_mat1_reshaped, 2)
                mini_mat1_means = torch.reshape(mini_mat1_means, [(mini_mat1_means.size())[0], (mini_mat1_means.size())[1], 1, 1])
                mini_mat1_means = mini_mat1_means.expand_as(mini_mat1)
                mini_mat0_centered = torch.add(mini_mat0, -1, mini_mat0_means)
                mini_mat1_centered = torch.add(mini_mat1, -1, mini_mat1_means)
                prod = torch.mul(mini_mat0_centered, mini_mat1_centered)
                mini_mat0_squared = torch.pow(mini_mat0_centered, 2)
                mini_mat1_squared = torch.pow(mini_mat1_centered, 2)
                mini_mat0_mat_var = torch.sqrt(torch.sum(mini_mat0_squared, (2, 3)))
                mini_mat1_mat_var = torch.sqrt(torch.sum(mini_mat1_squared, (2, 3)))
                cov = torch.sum(prod, (2, 3))
                correlation_coeff = torch.div(cov, torch.mul(mini_mat0_mat_var, mini_mat1_mat_var))
                both_consts = torch.mul(correlation_coeff != correlation_coeff, cov == 0)
                correlation_coeff[correlation_coeff != correlation_coeff] = 0
                # correlation_coeff[both_consts==1] = 1
                correlation_coeff_tot += (torch.sum(torch.abs(correlation_coeff), 1))
                # correlation_coeff_tot += (torch.sum(torch.max(correlation_coeff, torch.zeros_like(correlation_coeff)), 1))
        return correlation_coeff_tot

    ############################################################################################
    # Alternative 17 - Distance between sorted eigenvalues of each matrix
    ############################################################################################
    def Alt17(self,kk,flat0,flat1,outs0):
        eig0 = np.linalg.eig(flat0)[0]
        eig1 = np.linalg.eig(flat1)[0]
        cur_score = np.sum(np.linalg.norm(np.sort(eig0,2)-np.sort(eig1,2),2,2),1)
        return (torch.tensor(cur_score,dtype=torch.float64,device=torch.device('cuda:0'))).float()

    ############################################################################################
    # Alternative 18 - Distance between sum of eigen values, faster than Alt4
    ############################################################################################
    def Alt18(self,kk,flat0,flat1,outs0):
        eig0_sum = np.trace(flat0,0,2,3)
        eig1_sum = np.trace(flat1,0,2,3)
        cur_score = np.sum(np.abs(eig0_sum-eig1_sum),1)
        return (torch.tensor(cur_score,dtype=torch.float64,device=torch.device('cuda:0'))).float()

    ############################################################################################
    # Alternative 19 - A combination of 4 alternatives: cos_sim, L1 diff, and eigVals (sum and l2 norm)
    ############################################################################################
    def Alt19(self,kk, outs0, outs1):
        sum_res = self.Alt12(kk, outs0, outs1)
        # print("finished running alt. #1 of 4...")
        # cos_sim_score = (1. - util.cos_sim(outs0[kk], outs1[kk]))
        # print("finished running alt. #2 of 4...")
        eig_vals_sum = self.Alt18(kk,outs0[kk].cpu().data.numpy(),outs1[kk].cpu().data.numpy(),outs0)
        # print("finished running alt. #3 of 4...")
        eig_vals = self.Alt17(kk,outs0[kk].cpu().data.numpy(),outs1[kk].cpu().data.numpy(),outs0)
        # print("finished running alt. #4 of 4...")
        return torch.stack((sum_res, sum_res, eig_vals_sum,eig_vals), 1)

    ############################################################################################
    # Alternative 20 - Calculate eigen vectors and eigen values, then calculate the distance
    # between the two eigen vectors with the largest eigen values.
    # This is done once for the original feature, and once for the same feature transposed.
    ############################################################################################
    def Alt20(self,kk, outs0, outs1):

        outs0 = util.normalize_tensor(outs0[kk])
        outs1 = util.normalize_tensor(outs1[kk])
        outs0_reshaped = torch.reshape(outs0,[(outs0.size())[0],(outs0.size())[1],(outs0.size())[2]*(outs0.size())[3]])
        outs1_reshaped = torch.reshape(outs1,[(outs1.size())[0],(outs1.size())[1],(outs1.size())[2]*(outs1.size())[3]])
        outs0_means = torch.mean(outs0_reshaped,2)
        outs0_means = torch.reshape(outs0_means,[(outs0.size())[0],(outs0.size())[1],1,1])
        outs0_means = outs0_means.expand_as(outs0)
        outs0 = outs0 - outs0_means
        outs1_means = torch.mean(outs1_reshaped,2)
        outs1_means = torch.reshape(outs1_means,[(outs1.size())[0],(outs1.size())[1],1,1])
        outs1_means = outs1_means.expand_as(outs1)

        outs1 = outs1 - outs1_means
        outs0 = outs0.cpu().data.numpy()
        outs1 = outs1.cpu().data.numpy()
        eigVals0, eigVecs0 = np.linalg.eig(outs0)
        eigVals1, eigVecs1 = np.linalg.eig(outs1)
        eigVals0T, eigVecs0T = np.linalg.eig(np.transpose(outs0,(0,1,3,2)))
        eigVals1T, eigVecs1T = np.linalg.eig(np.transpose(outs1,(0,1,3,2)))
        maxInd0 = np.argmax(np.abs(eigVals0),2)
        maxInd1 = np.argmax(np.abs(eigVals1),2)
        maxInd0T = np.argmax(np.abs(eigVals0T),2)
        maxInd1T = np.argmax(np.abs(eigVals1T),2)
        cur_score = []
        for img in range ((outs0.shape)[0]):
            img_score = 0
            for feat in range ((outs0.shape)[1]):
                # feat_score = 0
                feat_score1 = np.linalg.norm(eigVecs0[img,feat,maxInd0[img,feat]] - eigVecs1[img,feat,maxInd1[img,feat]],2)
                feat_score2 = np.linalg.norm(eigVecs0T[img,feat,maxInd0T[img,feat]] - eigVecs1T[img,feat,maxInd1T[img,feat]],2)
                img_score += min(feat_score1,feat_score2)
            cur_score += [img_score]
        return (torch.tensor(cur_score, dtype=torch.float64, device=torch.device('cuda:0'))).float()

    ############################################################################################
    # Alternative 21 - PCA - Calculate sample-covariance mat and find its primary components.
    # This will be done for each feature and its transpose.
    ############################################################################################
    def Alt21(self,kk, outs0, outs1):

        outs0 = outs0[kk]
        outs1 = outs1[kk]
        outs0_row_means = torch.mean(outs0,3,True)
        outs0_row_means = outs0_row_means.expand_as(outs0)
        outs0_col_means = torch.mean(outs0,2,True)
        outs0_col_means = outs0_col_means.expand_as(outs0)
        outs1_row_means = torch.mean(outs1,3,True)
        outs1_row_means = outs1_row_means.expand_as(outs1)
        outs1_col_means = torch.mean(outs1,2,True)
        outs1_col_means = outs1_col_means.expand_as(outs1)
        outs0_row_centered = torch.add(outs0,-1,outs0_row_means)
        outs0_col_centered = torch.add(outs0,-1, outs0_col_means)
        outs1_row_centered = torch.add(outs1,-1,outs1_row_means)
        outs1_col_centered = torch.add(outs1,-1, outs1_col_means)
        samp_cov_rows0 = torch.matmul(torch.transpose(outs0_row_centered,2,3),outs0_row_centered)
        samp_cov_rows1 = torch.matmul(torch.transpose(outs1_row_centered,2,3),outs1_row_centered)
        samp_cov_cols0 = torch.matmul(outs0_col_centered,torch.transpose(outs0_col_centered,2,3))
        samp_cov_cols1 = torch.matmul(outs1_col_centered,torch.transpose(outs1_col_centered,2,3))
        samp_cov_rows0 = samp_cov_rows0.cpu().data.numpy()
        samp_cov_rows1 = samp_cov_rows1.cpu().data.numpy()
        samp_cov_cols0 = samp_cov_cols0.cpu().data.numpy()
        samp_cov_cols1 = samp_cov_cols1.cpu().data.numpy()
        eigValsRows0, eigVecsRows0 = np.linalg.eig(samp_cov_rows0)
        eigValsRows1, eigVecsRows1 = np.linalg.eig(samp_cov_rows1)
        eigValsCols0, eigVecsCols0 = np.linalg.eig(samp_cov_cols0)
        eigValsCols1, eigVecsCols1 = np.linalg.eig(samp_cov_cols1)
        maxIndRows0 = np.argmax(np.abs(eigValsRows0),2)
        maxIndRows1 = np.argmax(np.abs(eigValsRows1),2)
        maxIndCols0 = np.argmax(np.abs(eigValsCols0),2)
        maxIndCols1 = np.argmax(np.abs(eigValsCols1),2)
        cur_score = []
        for img in range ((outs0.size())[0]):
            img_score = 0
            for feat in range ((outs0.size())[1]):
                feat_score = 0
                feat_score += np.linalg.norm(eigVecsRows0[img,feat,maxIndRows0[img,feat]] - eigVecsRows1[img,feat,maxIndRows1[img,feat]],2)
                feat_score += np.linalg.norm(eigVecsCols0[img,feat,maxIndCols0[img,feat]] - eigVecsCols1[img,feat,maxIndCols1[img,feat]],2)
                img_score += feat_score/2
            cur_score += [img_score]
        return (torch.tensor(cur_score, dtype=torch.float64, device=torch.device('cuda:0'))).float()

    # ############################################################################################
    # # Alternative 22 - PCA - Treating each feature as H*W vector with C num of samples of this vector
    # ############################################################################################
    # def Alt22(self,kk, outs0, outs1):
    #
    #     outs0 = util.normalize_tensor(outs0[kk])
    #     outs1 = util.normalize_tensor(outs1[kk])
    #     outs0_reshaped = torch.reshape(outs0,[(outs0.size())[0],(outs0.size())[1],(outs0.size())[2]*(outs0.size())[3]])
    #     outs1_reshaped = torch.reshape(outs1,[(outs1.size())[0],(outs1.size())[1],(outs1.size())[2]*(outs1.size())[3]])
    #     outs0_means = torch.mean(outs0_reshaped,1)
    #     outs0_reshaped = outs0_reshaped - (torch.reshape(outs0_means,[(outs0.size())[0],1,(outs0_reshaped.size())[2]])).expand_as(outs0_reshaped)
    #     outs1_means = torch.mean(outs1_reshaped,1)
    #     outs1_reshaped = outs1_reshaped - (torch.reshape(outs1_means,[(outs1.size())[0],1,(outs1_reshaped.size())[2]])).expand_as(outs1_reshaped)
    #     outs0 = torch.matmul(torch.transpose(outs0_reshaped,1,2),outs0_reshaped)
    #     outs1 = torch.matmul(torch.transpose(outs1_reshaped,1,2),outs1_reshaped)
    #     outs0 = outs0.cpu().data.numpy()
    #     outs1 = outs1.cpu().data.numpy()
    #     eigVals0, eigVecs0 = np.linalg.eig(outs0)
    #     eigVals1, eigVecs1 = np.linalg.eig(outs1)
    #     maxInd0 = np.argmax(np.abs(eigVals0),1)
    #     maxInd1 = np.argmax(np.abs(eigVals1),1)
    #     cur_score = []
    #     for img in range ((outs0.shape)[0]):
    #         img_score = np.linalg.norm(eigVecs0[img,maxInd0[img]] - eigVecs1[img,maxInd1[img]],2)
    #         cur_score += [img_score]
    #     return (torch.tensor(cur_score, dtype=torch.float64, device=torch.device('cuda:0'))).float()

    def forward(self, in0, in1, retPerLayer=False):
        is_median = False
        alt = self.alt
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)
        res_arr = []
        if(retPerLayer):
            all_scores = []
        img = 0 #temp

        # for img in range (((outs0[0]).size())[0]):
        for (kk,out0) in enumerate(outs0):
            num_img = ((outs0[kk]).size())[0]  # for batch size != 1
            ten = torch.ones((num_img,), dtype=torch.float64)
            if is_median:
                temp_arr = outs0[kk].cpu().data.numpy()
                sz = (temp_arr.size())[2]
                flat0 = signal.medfilt(temp_arr,[1,1,3,3])[0,:,1:sz-1,1:sz-1]
                temp_arr = outs1[kk].cpu().data.numpy()
                flat1 = signal.medfilt(temp_arr,[1,1,3,3])[0,:,1:sz-1,1:sz-1]
            else:
                flat0 = (util.normalize_tensor(outs0[kk])).cpu().data.numpy()
                flat1 = (util.normalize_tensor(outs1[kk])).cpu().data.numpy()

            if alt == "Alt1":
                cur_score = self.Alt1(kk,flat0,flat1,outs0)
            elif alt == "Alt2":
                cur_score = self.Alt2(kk, flat0, flat1, outs0,img)
            elif alt == "Alt3":
                cur_score = self.Alt3(kk, flat0, flat1, outs0,outs1, img)
            elif alt == "Alt4":
                cur_score = self.Alt4(kk, outs0[kk].cpu().data.numpy(), outs1[kk].cpu().data.numpy(), outs0, img)
            elif alt == "Alt5":
                cur_score = self.Alt5(kk, flat0, flat1, outs0, img)
            elif alt == "Alt6":
                cur_score = self.Alt6(kk, flat0, flat1, outs0, img)
            elif alt == "Alt7":
                cur_score = self.Alt7(kk, flat0, flat1, outs0, img)
            elif alt == "Alt8":
                cur_score = self.Alt8(kk, flat0, flat1, outs0, img)
            elif alt == "Alt9":
                cur_score = self.Alt9(kk, outs0, outs1)
            elif alt == "Alt10":
                cur_score = self.Alt10(kk, outs0, outs1)
            elif alt == "Alt11":
                cur_score = self.Alt11(kk, outs0, outs1)
            elif alt == "Alt12":
                cur_score = self.Alt12(kk, outs0, outs1)
            elif alt == "Alt13":
                cur_score = self.Alt13(kk, outs0, outs1)
            elif alt == "Alt14":
                cur_score = self.Alt14(kk, outs0, outs1)
            elif alt == "Alt15":
                cur_score = self.Alt15(kk, outs0, outs1)
            elif alt == "Alt16":
                cur_score = self.Alt16(kk, outs0, outs1)
            elif alt == "Alt17":
                cur_score = self.Alt17(kk, outs0[kk].cpu().data.numpy(), outs1[kk].cpu().data.numpy(), outs0)
            elif alt == "Alt18":
                cur_score = self.Alt18(kk, outs0[kk].cpu().data.numpy(), outs1[kk].cpu().data.numpy(), outs0)
            elif alt == "Alt19":
                cur_score = self.Alt19(kk, outs0, outs1)
            elif alt == "Alt20":
                cur_score = self.Alt20(kk, outs0, outs1)
            elif alt == "Alt21":
                cur_score = self.Alt21(kk, outs0, outs1)
            elif alt == "Alt22":
                cur_score = self.Alt22(kk, outs0, outs1)

                #############################################################################################
                # Alternative 1 - Create covariance matrix of 2 features, calc the determinant and accumulate
                # Result tensor
                # ten = torch.ones((1,), dtype=torch.float64)
                # # Normalization
                # flat0 = (util.normalize_tensor(outs0[kk])).cpu().data.numpy()
                # flat1 = (util.normalize_tensor(outs1[kk])).cpu().data.numpy()
                # cur_score = 0
                # for idx in range(((outs0[kk]).size())[1]):
                #     # print(idx)
                #     vec0 = np.concatenate(flat0[0,idx,:,:])
                #     vec1 = np.concatenate(flat1[0,idx,:,:])
                #     covmat = np.cov(vec0,vec1)
                #     cur_score += np.linalg.det(covmat)
                    # print('covmat: {0}'.format(covmat))
                ############################################################################################
                # Alternative 2 - Create covariance matrix of 2 features, take only the covariance value
                # (cell (2,1) in the cov matrix) and accumulate
                # Alt2 = 1
                # num_img = ((outs0[kk]).size())[0] # for batch size != 1
                # ten = torch.ones((num_img,), dtype=torch.float64)
                # # Normalization
                # flat0 = (util.normalize_tensor(outs0[kk])).cpu().data.numpy()
                # flat1 = (util.normalize_tensor(outs1[kk])).cpu().data.numpy()
                # cur_score = 0
                # for img in range (((outs0[kk]).size())[0]):
                #     for idx in range(((outs0[kk]).size())[1]):
                #         # print(idx)
                #         vec0 = np.concatenate(flat0[img,idx,:,:])
                #         vec1 = np.concatenate(flat1[img,idx,:,:])
                #         covmat = np.cov(vec0,vec1)
                #         if covmat[0, 0]!=0 and covmat[1, 1] != 0:
                #             cur_score += np.abs(covmat[1, 0]/np.sqrt(covmat[1, 1] * covmat[0, 0])) # Absolute value of correlation coefficient
                #         elif covmat[0, 0]==0 and covmat[1, 1] == 0:  # Two constants -> high correlation
                #             cur_score += 1
                #     print('covmat: {0}'.format(covmat))
                ############################################################################################
                ############################################################################################
                # Alternative 3 - Use the original method (cos_sim) and correlation coefficient
                # Alt3 = 1
                # num_img = ((outs0[kk]).size())[0] # for batch size != 1
                #
                # # Normalization
                # # flat0 = (util.normalize_tensor(outs0[kk])).cpu().data.numpy()
                # # flat1 = (util.normalize_tensor(outs1[kk])).cpu().data.numpy()
                # flat0 = outs0[kk].cpu().data.numpy()
                # flat1 = outs1[kk].cpu().data.numpy()
                # cur_score = 0
                # for img in range(((outs0[kk]).size())[0]):
                #     for idx in range(((outs0[kk]).size())[1]):
                #         # print(idx)
                #         vec0 = np.concatenate(flat0[img,idx,:,:])
                #         vec1 = np.concatenate(flat1[img,idx,:,:])
                #         vec0 = (vec0-np.mean(vec0))/(np.sqrt(np.cov(vec0)+1e-5))
                #         vec1 = (vec1-np.mean(vec1))/(np.sqrt(np.cov(vec1)+1e-5))
                #         covmat = np.cov(vec0,vec1)
                #         if covmat[0, 0] != 0 and covmat[1, 1] != 0:
                #             cur_score += 1. - (covmat[1, 0]/np.sqrt(covmat[1, 1] * covmat[0, 0])) # Absolute value of correlation coefficient
                #         elif covmat[0, 0] == 0 and covmat[1, 1] == 0:  # Two constants -> high correlation
                #             cur_score += 0
                #         else:
                #             cur_score += 1
                # cur_score_org = (1.-util.cos_sim(outs0[kk],outs1[kk]))
                #     print('covmat: {0}'.format(covmat))
                ############################################################################################
                # Alternative 4 - Sum and compare each feature's matrix Eigenvalues
                ############################################################################################
                # Alt4 = 1
                # flat0 = outs0[kk].cpu().data.numpy()
                # flat1 = outs1[kk].cpu().data.numpy()
                # cur_score = 0
                # for img in range(((outs0[kk]).size())[0]):
                #     for idx in range(((outs0[kk]).size())[1]):
                #         eig_sum0 = np.linalg.eig(flat0[img,idx,:,:])[0].sum()
                #         eig_sum1 = np.linalg.eig(flat1[img,idx,:,:])[0].sum()
                #         cur_score += np.abs(eig_sum0 - eig_sum1)
                ############################################################################################
                # Original Code
                # cur_score = (1.-util.cos_sim(outs0[kk],outs1[kk]))
                ############################################################################################
            if(kk==0):
                val = 1.*cur_score
            else:
                val = val + cur_score

        if(retPerLayer):
            all_scores+=[cur_score]
        if alt in ["Alt6", "Alt5", "Alt4", "Alt3"]:
            res_arr += [val]
        elif alt in ["Alt8", "Alt7", "Alt2"]:
            res_arr += [1/val]
        elif alt in ["Alt1"]:
            res_arr += 1-val
        # if alt in ["Alt6","Alt5","Alt4","Alt3"]:
        #     val = ten.new_tensor((1,), val, dtype=torch.float64)
        # elif alt in ["Alt8","Alt7","Alt2"]:
        #     val = ten.new_full((1,), 1/val, dtype=torch.float64)
        # elif alt in ["Alt1"]:
        #     val = ten.new_full((1,), 1-val, dtype=torch.float64)

        if alt in ["Alt9","Alt10","Alt16"]:
            return torch.reciprocal(val)
        elif alt in ["Alt11","Alt12","Alt13","Alt14","Alt17","Alt18","Alt19","Alt20","Alt21","Alt22"]:
            return val
        if(retPerLayer):
            return (ten.new_tensor(res_arr,dtype=torch.float64), all_scores)
        else:
            return ten.new_tensor(res_arr,dtype=torch.float64)

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, use_gpu=True, spatial=False, version='0.1'):
        super(PNetLin, self).__init__()

        self.use_gpu = use_gpu
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.version = version

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]

        if(self.pnet_tune):
            self.net = net_type(pretrained=not self.pnet_rand,requires_grad=True)
        else:
            self.net = [net_type(pretrained=not self.pnet_rand,requires_grad=False),]

        self.lin0 = NetLinLayer(self.chns[0],use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1],use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2],use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3],use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4],use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5],use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6],use_dropout=use_dropout)
            self.lins+=[self.lin5,self.lin6]

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))

        if(use_gpu):
            if(self.pnet_tune):
                self.net.cuda()
            else:
                self.net[0].cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()
            if(self.pnet_type=='squeeze'):
                self.lin5.cuda()
                self.lin6.cuda()

    def forward(self, in0, in1):
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        if(self.version=='0.0'):
            # v0.0 - original release had a bug, where input was not scaled
            in0_input = in0
            in1_input = in1
        else:
            # v0.1
            in0_input = in0_sc
            in1_input = in1_sc

        if(self.pnet_tune):
            outs0 = self.net.forward(in0_input)
            outs1 = self.net.forward(in1_input)
        else:
            outs0 = self.net[0].forward(in0_input)
            outs1 = self.net[0].forward(in1_input)

        feats0 = {}
        feats1 = {}
        diffs = [0]*len(outs0)

        for (kk,out0) in enumerate(outs0):
            feats0[kk] = util.normalize_tensor(outs0[kk])
            feats1[kk] = util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if self.spatial:
            lin_models = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if(self.pnet_type=='squeeze'):
                lin_models.extend([self.lin5, self.lin6])
            res = [lin_models[kk].model(diffs[kk]) for kk in range(len(diffs))]
            return res
			
        val = torch.mean(torch.mean(self.lin0.model(diffs[0]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin1.model(diffs[1]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin2.model(diffs[2]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin3.model(diffs[3]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin4.model(diffs[4]),dim=3),dim=2)
        if(self.pnet_type=='squeeze'):
            val = val + torch.mean(torch.mean(self.lin5.model(diffs[5]),dim=3),dim=2)
            val = val + torch.mean(torch.mean(self.lin6.model(diffs[6]),dim=3),dim=2)

        val = val.view(val.size()[0],val.size()[1],1,1)

        return val

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32,use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, use_gpu=True, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.use_gpu = use_gpu
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()
        self.model = nn.Sequential(*[self.net])

        if(self.use_gpu):
            self.net.cuda()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        if(self.use_gpu):
            per = per.cuda()
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

class L2(FakeNet):

    def forward(self, in0, in1):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = util.dssim(1.*util.tensor2im(in0.data), 1.*util.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
