import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from Active_shape_model import *
from utils.data_loading import BasicDataset
from pathlib import Path
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from fusion_net.net import SPGNet
from evaluate import evaluate1
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from utils.cluster import custom_distance,total_procrustes_analysis
from sklearn.cluster import AgglomerativeClustering
import re
    

train_dir_img = Path('./dataset/train/imgs/')
train_dir_mask = Path('./dataset/train/masks/')
train_dir_lmk = Path('./dataset/train/lmks/')

val_dir_img = Path('./dataset/train/imgs/')
val_dir_mask = Path('./dataset/train/masks/')
val_dir_lmk = Path('./dataset/train/lmks/')


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):  #  每个batch   h,w
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)   # input[:, channel, ...].shape:   bs,h,w

    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def load_landmarks(lmk_name_list, number_points): 
    if number_points == 128:
        inputList = np.empty([len(lmk_name_list), 129, 2])
    elif number_points == 64:
        inputList = np.empty([len(lmk_name_list), 65, 2])
    i = 0
    idlist = np.empty([len(lmk_name_list)])
    # Use slicing to extract the content between 'a' and 'b'
    for inputName in lmk_name_list:
        # print(inputName)
        file_name = os.path.basename(inputName)
        parts = file_name.split('.pts')
        id = parts[0]
        points, num_pts, width, height = read_pts_file(inputName)
        points.append([width, height])
        # print(id)
        # print(len(points))
        inputList[i] = points
        idlist[i] =  "".join(list(filter(str.isdigit, id)))
        i += 1
        # print(inputList.shape)
    idlist = idlist.astype(int)
    return inputList, idlist

def select_best_k(procrustes_distance_matrix):
    
    max_var = 0
    k = 0
    for i in range(100):
        agg_clustering = AgglomerativeClustering(n_clusters=i+2, affinity='precomputed', linkage='complete', distance_threshold=None, compute_full_tree=True)
        cluster_labels = agg_clustering.fit_predict(procrustes_distance_matrix)
        
        # Calculate the center point of each cluster
        unique_labels = np.unique(cluster_labels)
        cluster_centers = []
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            cluster_points = data[cluster_labels == label]
            # print(cluster_points.shape)
            cluster_points = total_procrustes_analysis(cluster_points)  # Align and standardize the cluster
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        cluster_centers = np.array(cluster_centers)
        
        shapes = cluster_centers
        distances = []
        for i in range(len(shapes)):
            for j in range(i+1, len(shapes)):
                distance = custom_distance(shapes[i], shapes[j])
                distances.append(distance)
        
        # Calculate the variance of Procrustes distances
        distance_variance = np.var(distances)
        print("k ="+str(i+2)+':'+str(distance_variance))
            
        if distance_variance >= max_var:
            max_var = distance_variance
            k = i+2
    print("k = "+str(k)+" has max variance:"+str(max_var))
    return k

def generate_shapes(k, data, points, idlist, procrustes_distance_matrix, number_point):
    agg_clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete', distance_threshold=None, compute_full_tree=True)

    # Perform clustering
    cluster_labels = agg_clustering.fit_predict(procrustes_distance_matrix)
    
    # Calculate the center point of each cluster
    unique_labels = np.unique(cluster_labels)
    cluster_centers = []

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = data[cluster_labels == label]

        cluster_points = total_procrustes_analysis(cluster_points)  # Align and standardize the cluster
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)

    cluster_centers = np.array(cluster_centers)
    shape_list = []
    shape_id_list = []
    max_t = 0
    
    # Calculate the maximum t
    for label in unique_labels:
        if label == -1:
            continue
        cluster_data = points[cluster_labels == label]  # cluster_num, 258 including two size data
        cluster_id = idlist[cluster_labels == label]
        #print(cluster_data.shape)
        #print(cluster_id.shape)
        shapes = PointsReader_cluster.read_points(cluster_data, number_point)
        asm_model = ActiveShapeModel(shapes, t_max=0, padding=False)
        mean_shape = asm_model.mean.reshape(1, -1, 2)
        evec = asm_model.evecs.transpose(1, 0).reshape(asm_model.modes, -1, 2)
        if asm_model.modes > max_t:
            max_t = asm_model.modes
    
    # Assign the actual maximum t
    for label in unique_labels:
        if label == -1:
            continue
        cluster_data = points[cluster_labels == label]  # cluster_num, 258 including two size data
        cluster_id = idlist[cluster_labels == label]
        shapes = PointsReader_cluster.read_points(cluster_data, number_point)
        asm_model = ActiveShapeModel(shapes, t_max=max_t, padding=True)
        mean_shape = asm_model.mean.reshape(1, -1, 2)
        evec = asm_model.evecs.transpose(1, 0).reshape(asm_model.modes, -1, 2)
        shape = np.concatenate((mean_shape, evec), axis=0)
        shape_list.append(shape)
        shape_id_list.append(cluster_id)

    shape_list = np.array(shape_list)

    return shape_list, shape_id_list

def find_max_numbered_file(folder_path):
    max_number = float('-inf')  # Initially set to negative infinity
    max_numbered_file_path = None

    # Traverse all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Use regular expression to extract the number part from the filename
        match = re.search(r'(\d+)', filename)
        
        # If a number is found, compare it
        if match:
            current_number = int(match.group())
            if current_number > max_number:
                max_number = current_number
                max_numbered_file_path = file_path

    return max_numbered_file_path

def train(train_set,val_set,shapes,shapes_id,number_points):
    batch_size = 16
    epochs = 150
    learning_rate = 1e-4
    save_checkpoint = True
    max_dice = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dir_checkpoint = Path('./checkpoint')

    # Create dataloaders
    loader_args = dict(batch_size=batch_size, num_workers=32, pin_memory=True)
    loader_args_val = dict(batch_size=1, num_workers=32, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True,drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args_val)
    
    n_train = len(train_loader)
    n_val = len(val_loader)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {1}
    ''')
    
    # Create Network
    
    shapes = torch.from_numpy(shapes).permute(1,2,3,0).to(device=device)   
    shape_prior = shapes.detach()    #.requires_grad_(False)
     
    net = SPGNet(in_channels = 3,n_classes = 2,img_size=256,shape_nclasses = shapes.size(3),n_evecs=shapes.size(0),shape_prior = shape_prior,number_point = number_points)
    net.to(device=device)
    
    # Set up the optimizer, the loss, the learning rate scheduler
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)

    criterion_lmks = nn.L1Loss()
    criterion_lmks_fine = nn.L1Loss()
    criterion_class = nn.BCELoss()
    criterion_segmap = nn.CrossEntropyLoss()
    
    lambda1 = 1
    lambda2 = 1
    lambda3 = 1
    lambda4 = 1
    lambda5 = 1
    lambda6 = 1
    global_step = 0
    
    # Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='atch') as pbar:
            for batch in train_loader:
                
                images = batch['image']
                true_masks = batch['mask']
                true_lmks = batch['lmk']
                numbers = batch['id_number']
                
                
                shape_class = torch.zeros(len(numbers), len(shapes_id))

                for bs in range(len(numbers)):
                    for i in range(len(shapes_id)):
                        if numbers[bs] in shapes_id[i]:
                            shape_class[bs][i] = 1


                true_shape_class = shape_class.float()

                images = images.to(device=device, dtype=torch.float32)
                true_lmks = true_lmks.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_shape_class = true_shape_class.to(device=device, dtype=torch.float32)

                shape_class_pred,lmks_pred,fine_lmks,segmap= net(images)
                loss1 = lambda1*criterion_class(shape_class_pred,true_shape_class)
                loss2 = lambda2*criterion_lmks(lmks_pred,true_lmks) 
                loss3 = lambda3*criterion_lmks_fine(fine_lmks[-3],true_lmks)
                loss4 = lambda4*criterion_lmks_fine(fine_lmks[-2],true_lmks)
                loss5 = lambda5*criterion_lmks_fine(fine_lmks[-1],true_lmks)
                loss6 = lambda6*(criterion_segmap(segmap,true_masks) + dice_loss(F.softmax(segmap, dim=1).float(),F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float(),multiclass=True) )
                loss = loss1+loss2+loss3+loss4+loss5+loss6
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                division_step = len(train_loader)

                if global_step % division_step == 0:
                    fine_mean_dice_score = 0
                    segmap_mean_dice_score,fine_mean_dice_score = evaluate1(net, val_loader, device)
                    logging.info('Validation fine Dice score: {}'.format(fine_mean_dice_score))
                           
        if save_checkpoint and fine_mean_dice_score>max_dice: 
            max_dice =  fine_mean_dice_score 
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        scheduler.step()
    print("max_dice = "+str(max_dice))         
                      
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    number_points = 128
    
    train_set = BasicDataset(train_dir_img, train_dir_mask,train_dir_lmk, 1, shapes_id = None)
    val_set = BasicDataset(val_dir_img, val_dir_mask,val_dir_lmk, 1, shapes_id = None)
    lmk_dir = str(train_dir_lmk)

    lmk_dir_name = []
    for i in range(len(train_set)):
        lmk_name = str(lmk_dir)+'/'+train_set[i]['id']+'_lmk.pts'
        lmk_dir_name.append(lmk_name)
    points,idlist = load_landmarks(lmk_dir_name,number_points=number_points)
    points = points.reshape(points.shape[0], -1)
    if number_points == 64:
        data = points[:, :128]
    elif number_points == 128:
        data = points[:, :256]
        
    procrustes_distance_matrix = pairwise_distances(data, metric=custom_distance)
    k = select_best_k(procrustes_distance_matrix)
    shapes,shapes_id = generate_shapes(k,data,points,idlist,procrustes_distance_matrix,number_point=number_points)
    train(train_set,val_set,shapes,shapes_id,number_points)
      