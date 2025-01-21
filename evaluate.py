import torch
import torch.nn.functional as F
from tqdm import tqdm
import surface_distance as surfdist
import numpy as np
from torch.utils.data import DataLoader,random_split,ConcatDataset
import cv2
import math
# from utils.dice_score import multiclass_dice_coeff, dice_coeff

def pixel_accuracy(input, target):
    """
    input: torch.FloatTensor:(N, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 3
    assert len(target.size()) == 3
    N, H, W = target.size()
    
    # input = F.softmax(input, dim=1)
    # arg_max = torch.argmax(input, dim=1)
    # (TP + TN) / (TP + TN + FP + FN)
    return torch.sum(input == target) / (N * H * W)

def jaccard(predict, target):
    jac = 0
    predict  =  predict.type(torch.IntTensor)
    target   = target.type(torch.IntTensor)
    for i in range(predict.shape[0]):
        intersection = torch.sum(predict[i,...]&target[i,...])
        union = torch.sum(predict[i,...]|target[i,...])
        jac += intersection / union
    return jac/predict.shape[0]

def dice_coeff(pred_mask, true_mask):
    intersection = torch.sum(pred_mask * true_mask)
    prediction = torch.sum(pred_mask)
    ground_truth = torch.sum(true_mask)
    
    dice = (2 * intersection) / (prediction + ground_truth)
    
    return dice.item()

def evaluate1(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)   
    dice_score = 0
    fine_dice_score = 0
    segmap_dice_score = 0

    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        img_w ,img_h = batch['width'],batch['height']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float)
        img_w = img_w.to(device=device, dtype=torch.int)
        img_h = img_h.to(device=device, dtype=torch.int)

        with torch.no_grad():
            # predict the mask
            shape_class_pred,lmks_pred,fine_lmks,segmap= net(image)
            segmap_pred = segmap.argmax(dim=1)
            fine_lmks[-1][0][:,0] = (256)*fine_lmks[-1][0][:,0]  
            fine_lmks[-1][0][:,1] = (256)*fine_lmks[-1][0][:,1]  
            points = fine_lmks[-1][0].int().tolist()
            
            img = torch.zeros(1, 256, 256, dtype=torch.uint8)
            img_np = img.numpy().squeeze()
            points_np = np.array([points], dtype=np.int32)
            cv2.fillPoly(img_np, points_np, 255)
            img = torch.from_numpy(img_np/255).unsqueeze(0).cuda()
            

            
            segmap_dice_score += dice_coeff(segmap_pred, mask_true)
            
            fine_dice_score += dice_coeff(img, mask_true)
            




    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    #mean_dice_score = dice_score / num_val_batches
    fine_mean_dice_score = fine_dice_score / num_val_batches
    segmap_mean_dice_score = segmap_dice_score / num_val_batches

    return segmap_mean_dice_score,fine_mean_dice_score

def evaluate2(net, dataloader, device):
    net.eval()

    num_val_batches = len(dataloader)   #返回一个dataloader有多少个batch
    dice_score = 0
    ASD = 0
    HD95 = 0
    JAC = 0
    ASD_fine = 0
    HD95_fine = 0
    JAC_fine = 0
    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        img_w ,img_h = batch['width'],batch['height']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float)
        img_w = img_w.to(device=device, dtype=torch.int)
        img_h = img_h.to(device=device, dtype=torch.int)

        with torch.no_grad():
            # predict the mask
            # segmap_pred,lmk_pred, fine_lmk_pred = net(image)[-3:]
            shape_class_pred,lmks_pred,fine_lmks,segmap= net(image)
            segmap_pred = segmap.argmax(dim=1)
            
            
            fine_lmks[-1][0][:,0] = (img_w)*fine_lmks[-1][0][:,0]   
            fine_lmks[-1][0][:,1] = (img_h)*fine_lmks[-1][0][:,1] 
            points = fine_lmks[-1][0].int().tolist()
            
            img = torch.zeros(1, img_w, img_h, dtype=torch.uint8)
            img_np = img.numpy().squeeze()
            points_np = np.array([points], dtype=np.int32)
            cv2.fillPoly(img_np, points_np, 255)
            img = torch.from_numpy(img_np/255).unsqueeze(0).cuda()

            JAC += jaccard(mask_true, segmap_pred)
            JAC_fine += jaccard(mask_true, img)


            mask_true_ASD_HD95 = np.array(mask_true.cpu(), dtype= bool)
            mask_pred_ASD_HD95 = np.array(segmap_pred.cpu(), dtype= bool)
            fine_pred_ASD_HD95 = np.array(img.cpu(), dtype= bool)
            
            
            surface_distances = surfdist.compute_surface_distances(mask_true_ASD_HD95[0], mask_pred_ASD_HD95[0], spacing_mm=(1.0, 1.0))  
            fine_surface_distances = surfdist.compute_surface_distances(mask_true_ASD_HD95[0], fine_pred_ASD_HD95[0], spacing_mm=(1.0, 1.0))  
            
            
            #只计算二分类ASD
            avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
            avg_surf_dist = (avg_surf_dist[0]+avg_surf_dist[1])/2   
            if math.isnan(avg_surf_dist):
                continue
            ASD +=  avg_surf_dist
            
            avg_surf_dist_fine = surfdist.compute_average_surface_distance(fine_surface_distances)
            avg_surf_dist_fine = (avg_surf_dist_fine[0]+avg_surf_dist_fine[1])/2   
            if math.isnan(avg_surf_dist_fine):
                continue
            ASD_fine +=  avg_surf_dist_fine
            
            
            #只计算二分类HD95
            hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
            HD95 +=  hd_dist_95
            
            hd_dist_95_fine = surfdist.compute_robust_hausdorff(fine_surface_distances, 95)
            HD95_fine +=  hd_dist_95_fine

            


    # net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    JAC = JAC/num_val_batches
    ASD = ASD/num_val_batches
    HD95 = HD95/num_val_batches
    JAC_fine = JAC_fine/num_val_batches
    ASD_fine = ASD_fine/num_val_batches
    HD95_fine = HD95_fine/num_val_batches
    return JAC,ASD,HD95,JAC_fine,ASD_fine,HD95_fine
