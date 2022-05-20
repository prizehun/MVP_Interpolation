from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch 
import cv2 
import math
import os
import warnings 
H=900
W=1600


class PaintDataSet(Dataset):
    def __init__(
        self,
        info_path,
        predictor
    ):
        infos = get_obj(info_path)
        sweeps = []

        paths = set()
        for info in infos:
            if info['lidar_path'] not in paths:
                paths.add(info['lidar_path'])
                sweeps.append(info)

            for sweep in info['sweeps']:
                if sweep['lidar_path'] not in paths: 
                    sweeps.append(sweep)
                    paths.add(sweep['lidar_path'])

        self.sweeps = sweeps
        self.predictor = predictor

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]
        tokens = info['lidar_path'].split('/')
        output_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        if os.path.isfile(output_path):
            return [] 

        all_cams_path = info['all_cams_path']

        all_data = [info]
        for path in all_cams_path:
            original_image = cv2.imread(path)
            
            if self.predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2] # 0 to 1, 2(excluded)
            image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = {"image": image, "height": height, "width": width}
            
            all_data.append(inputs) 

        return all_data 
    
    def __len__(self):
        return len(self.sweeps)

def is_within_mask(points_xyc, masks, H=900, W=1600):
    seg_mask = masks[:, :-1].reshape(-1, W, H) # -1 : automatical size, masks[:,0] to masks[:,-2]
    camera_id = masks[:, -1] # -1 : last element (column)
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0) 

@torch.no_grad()
def add_virtual_mask(masks, labels, points, raw_points, num_virtual=50, dist_thresh=3000, num_camera=6, intrinsics=None, transforms=None):
    points_xyc = points.reshape(-1, 5)[:, [0, 1, 4]] # x, y, z, valid_indicator, camera id , dist_thresh=3000

    valid = is_within_mask(points_xyc, masks)
    valid = valid * points.reshape(-1, 5)[:, 3:4]

    # remove camera id from masks 
    camera_ids = masks[:, -1]
    masks = masks[:, :-1]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )  

    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0 ).reshape(num_camera, -1).sum(dim=0).bool()

    offsets = [] 
    for mask in masks:
        indices = mask.reshape(W, H).nonzero()
        selected_indices = torch.randperm(len(indices), device=masks.device)[:num_virtual]
        if len(selected_indices) < num_virtual:
                selected_indices = torch.cat([selected_indices, selected_indices[
                    selected_indices.new_zeros(num_virtual-len(selected_indices))]])

        offset = indices[selected_indices]
        offsets.append(offset)
    
    offsets = torch.stack(offsets, dim=0)
    virtual_point_instance_ids = torch.arange(1, 1+masks.shape[0], 
        dtype=torch.float32, device='cuda:0').reshape(masks.shape[0], 1, 1).repeat(1, num_virtual, 1)

    virtual_points = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3)
    virtual_point_camera_ids = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1)

    valid_mask = valid.sum(dim=1)>0
    real_point_instance_ids = (torch.argmax(valid.float(), dim=1) + 1)[valid_mask]
    real_points = torch.cat([points_xyc[:, :2][valid_mask], real_point_instance_ids[..., None]], dim=-1)

    # avoid matching across instances 
    real_points[:, -1] *= 1e4 
    virtual_points[:, -1] *= 1e4 # to make m to mm

    if len(real_points) == 0:
        return None 
    #real_point = [0, x , y, instance_ids]
    #virtual = [offsets, 0, instance_ids]
    dist = torch.norm(virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1) #estimate distance, unsqueeze(0)->first channel add
    sorted_dist, sorted_indices = torch.sort(dist, dim=1)
    #nearest_dist, nearest_indices = torch.min(dist, dim=1) #search row
    #nearest_dist=sorted_dist[:,0]
    nearest_indices=sorted_indices[:,0]
    nearest_dist2=sorted_dist[:,1]
    nearest_indices2=sorted_indices[:,1]
    #nearest_dist3=sorted_dist[:,2]
    nearest_indices3=sorted_indices[:,2]
    #nearest_dist4=sorted_dist[:,3]
    nearest_indices4=sorted_indices[:,3]
    mask = nearest_dist2 < dist_thresh
    #mask2 = nearest_dist2 < dist_thresh
    #print(nearest_indices)
    indices = valid_mask.nonzero(as_tuple=False).reshape(-1)

    nearest_indices = indices[nearest_indices[mask]]
    nearest_indices2 = indices[nearest_indices2[mask]] # points in dist_thresh
    nearest_indices3 = indices[nearest_indices3[mask]]
    nearest_indices4 = indices[nearest_indices4[mask]]
    virtual_points = virtual_points[mask]
    #virtual_points2 = virtual_points[mask2]
    #print(nearest_indices)
    virtual_point_camera_ids = virtual_point_camera_ids[mask]
    #virtual_point_camera_ids2 = virtual_point_camera_ids[mask2]

    #interpolation mask 
    #mask2 = nearest_dist4 - nearest_dist < 2

    #nearest_indices = indices[nearest_indices[mask2]]
    #nearest_indices2 = indices[nearest_indices2[mask2]]
    #nearest_indices3 = indices[nearest_indices3[mask2]]
    #nearest_indices4 = indices[nearest_indices4[mask2]]
    #virtual_points = virtual_points[mask2]
    #virtual_point_camera_ids = virtual_point_camera_ids[mask2]

    all_virtual_points = [] 
    all_real_points = [] 
    all_point_labels = []

    #all_virtual_labels=[] #my code to get virtual labels

    torch.cuda.empty_cache()

    for i in range(num_camera):
        camera_mask = (virtual_point_camera_ids == i).squeeze()
        per_camera_virtual_points = virtual_points[camera_mask]
        per_camera_indices = nearest_indices[camera_mask]
        per_camera_indices2 = nearest_indices2[camera_mask]
        per_camera_indices3 = nearest_indices3[camera_mask]
        per_camera_indices4 = nearest_indices4[camera_mask]

        #print(point_labels[per_camera_indices])
        #print(point_labels[per_camera_indices2])
        #print(point_labels[per_camera_indices3])
        #print(point_labels[per_camera_indices4])

        #print(per_camera_indices.size())
        #print(per_camera_indices2.size())
        #print(per_camera_indices3.size())
        #print(per_camera_indices4.size())
        
        z1=points.reshape(-1, 5)[per_camera_indices, 2].reshape(1, -1)
        z2=points.reshape(-1, 5)[per_camera_indices2, 2].reshape(1, -1)
        z3=points.reshape(-1, 5)[per_camera_indices3, 2].reshape(1, -1)
        z4=points.reshape(-1, 5)[per_camera_indices4, 2].reshape(1, -1)

        z1=z1.tolist()
        z2=z2.tolist()
        z3=z3.tolist()
        z4=z4.tolist()
        
        xv=per_camera_virtual_points[:,0] #virtual point x
        yv=per_camera_virtual_points[:,1] #virtual point y

        xv=xv.tolist()
        yv=yv.tolist()
        #vp=per_camera_virtual_points[:, :2] #virtual point x, y
        
        x1=points.reshape(-1, 5)[per_camera_indices, 0].reshape(1, -1) #real point x1
        y1=points.reshape(-1, 5)[per_camera_indices, 1].reshape(1, -1) #real point y1
        x2=points.reshape(-1, 5)[per_camera_indices2, 0].reshape(1, -1) #real point x2
        y2=points.reshape(-1, 5)[per_camera_indices2, 1].reshape(1, -1) #real point y2
        x3=points.reshape(-1, 5)[per_camera_indices3, 0].reshape(1, -1) #real point x3
        y3=points.reshape(-1, 5)[per_camera_indices3, 1].reshape(1, -1) #real point y3
        x4=points.reshape(-1, 5)[per_camera_indices4, 0].reshape(1, -1) #real point x4
        y4=points.reshape(-1, 5)[per_camera_indices4, 1].reshape(1, -1) #real point y4

        #print(x1);print(x4);print(len(xv))

        x1=x1.tolist()
        x2=x2.tolist()
        x3=x3.tolist()
        x4=x4.tolist()
        y1=y1.tolist()
        y2=y2.tolist()
        y3=y3.tolist()
        y4=y4.tolist()

        z=[] #empty list
        pts2=np.float32([[0,1],[1,1],[1,0],[0,0]]) #transform 2d array
        #del_inx=[] #del list
        for j in range(len(x1[0])):
            cross=0
            m1=0;m2=0;n1=0;n2=0;temp_d=0;temp1=0;temp2=0;inx=0;pre_atX=0
            if (y1[0][j] > yv[j]) != (y2[0][j] > yv[j]):
                atX = (x2[0][j]-x1[0][j])*(yv[j]-y1[0][j])/(y2[0][j]-y1[0][j])+x1[0][j]
                if (xv[j] < atX) :
                    cross += 1
                    inx=1
            if (y2[0][j] > yv[j]) != (y3[0][j] > yv[j]):
                atX = (x3[0][j]-x2[0][j])*(yv[j]-y2[0][j])/(y3[0][j]-y2[0][j])+x2[0][j]
                if (xv[j] < atX) :
                    cross += 1
                    inx=2                
            if (y3[0][j] > yv[j]) != (y4[0][j] > yv[j]):
                atX = (x4[0][j]-x3[0][j])*(yv[j]-y3[0][j])/(y4[0][j]-y3[0][j])+x3[0][j]
                if (xv[j] < atX) :
                    cross += 1
                    inx=3                                                            
            if (y4[0][j] > yv[j]) != (y1[0][j] > yv[j]):
                atX = (x1[0][j]-x4[0][j])*(yv[j]-y4[0][j])/(y1[0][j]-y4[0][j])+x4[0][j]                    
                if (xv[j] < atX) :
                    cross += 1
                    inx=4
            if cross%2 > 0 :
                #dist_thr=math.sqrt((x1[0][j]-x4[0][j])**2 + (y1[0][j]-y4[0][j])**2 + (z1[0][j]-z4[0][j])**2)
                #print(dist_thr)
                #print(z1[0][j])
                #print(z4[0][j])
                #if dist_thr > 50 : #5cm?
                #    z.append((z1[0][j]+z2[0][j])/2)
                if abs(z1[0][j]-z4[0][j]) > 0.5: #50cm depth
                    z.append((z1[0][j]+z2[0][j])/2)
                elif inx==0 :
                    temp_d=(z1[0][j]+z2[0][j])/2
                    z.append(temp_d)
                else :                
                    if (inx==1) :
                        if (y1[0][j] > y2[0][j]) :  #right up y1
                            if (y3[0][j] > y4[0][j]) : #left up y3
                                pts1=np.float32([[x3[0][j],y3[0][j]], [x1[0][j],y1[0][j]], [x2[0][j],y2[0][j]], [x4[0][j],y4[0][j]]]) # start left up to clockwise
                                ztemp=[z3[0][j], z1[0][j], z2[0][j], z4[0][j]]
                            else :
                                pts1=np.float32([[x4[0][j],y4[0][j]], [x1[0][j],y1[0][j]], [x2[0][j],y2[0][j]], [x3[0][j],y3[0][j]]])
                                ztemp=[z4[0][j], z1[0][j], z2[0][j], z3[0][j]]
                        else :
                            if (y3[0][j] > y4[0][j]) : #left up y3
                                pts1=np.float32([[x3[0][j],y3[0][j]], [x2[0][j],y2[0][j]], [x1[0][j],y1[0][j]], [x4[0][j],y4[0][j]]]) # clockwise
                                ztemp=[z3[0][j], z2[0][j], z1[0][j], z4[0][j]]
                            else :
                                pts1=np.float32([[x4[0][j],y4[0][j]], [x2[0][j],y2[0][j]], [x1[0][j],y1[0][j]], [x3[0][j],y3[0][j]]])
                                ztemp=[z4[0][j], z2[0][j], z1[0][j], z3[0][j]]
                    if (inx==2) :
                        if (y2[0][j] > y3[0][j]) :  #right up y2
                            if (y1[0][j] > y4[0][j]) : #left up y1
                                pts1=np.float32([[x1[0][j],y1[0][j]], [x2[0][j],y2[0][j]], [x3[0][j],y3[0][j]], [x4[0][j],y4[0][j]]]) # clockwise
                                ztemp=[z1[0][j], z2[0][j], z3[0][j], z4[0][j]]
                            else :
                                pts1=np.float32([[x4[0][j],y4[0][j]], [x2[0][j],y2[0][j]], [x3[0][j],y3[0][j]], [x1[0][j],y1[0][j]]])
                                ztemp=[z4[0][j], z2[0][j], z3[0][j], z1[0][j]]
                        else :
                            if (y1[0][j] > y4[0][j]) : #left up y1
                                pts1=np.float32([[x1[0][j],y1[0][j]], [x3[0][j],y3[0][j]], [x2[0][j],y2[0][j]], [x4[0][j],y4[0][j]]]) # clockwise
                                ztemp=[z1[0][j], z3[0][j], z2[0][j], z4[0][j]]
                            else :
                                pts1=np.float32([[x4[0][j],y4[0][j]], [x3[0][j],y3[0][j]], [x2[0][j],y2[0][j]], [x1[0][j],y1[0][j]]])
                                ztemp=[z4[0][j], z3[0][j], z2[0][j], z1[0][j]]
                    if (inx==3) :
                        if (y3[0][j] > y4[0][j]) :  #right up y3
                            if (y1[0][j] > y2[0][j]) : #left up y1
                                pts1=np.float32([[x1[0][j],y1[0][j]], [x3[0][j],y3[0][j]], [x4[0][j],y4[0][j]], [x2[0][j],y2[0][j]]]) # clockwise
                                ztemp=[z1[0][j], z3[0][j], z4[0][j], z2[0][j]]
                            else :
                                pts1=np.float32([[x2[0][j],y2[0][j]], [x3[0][j],y3[0][j]], [x4[0][j],y4[0][j]], [x1[0][j],y1[0][j]]])
                                ztemp=[z2[0][j], z3[0][j], z4[0][j], z1[0][j]]
                        else :
                            if (y1[0][j] > y2[0][j]) : #left up y1
                                pts1=np.float32([[x1[0][j],y1[0][j]], [x4[0][j],y4[0][j]], [x3[0][j],y3[0][j]], [x2[0][j],y2[0][j]]]) # clockwise
                                ztemp=[z1[0][j], z4[0][j], z3[0][j], z2[0][j]]
                            else :
                                pts1=np.float32([[x2[0][j],y2[0][j]], [x4[0][j],y4[0][j]], [x3[0][j],y3[0][j]], [x1[0][j],y1[0][j]]])
                                ztemp=[z2[0][j], z4[0][j], z3[0][j], z1[0][j]]  
                    if (inx==4) :
                        if (y4[0][j] > y1[0][j]) :  #right up y4
                            if (y2[0][j] > y3[0][j]) : #left up y2
                                pts1=np.float32([[x2[0][j],y2[0][j]], [x4[0][j],y4[0][j]], [x1[0][j],y1[0][j]], [x3[0][j],y3[0][j]]]) # clockwise
                                ztemp=[z2[0][j], z4[0][j], z1[0][j], z3[0][j]]
                            else :
                                pts1=np.float32([[x3[0][j],y3[0][j]], [x4[0][j],y4[0][j]], [x1[0][j],y1[0][j]], [x2[0][j],y2[0][j]]])
                                ztemp=[z3[0][j], z4[0][j], z1[0][j], z2[0][j]]
                        else :
                            if (y2[0][j] > y3[0][j]) : #left up y1
                                pts1=np.float32([[x2[0][j],y2[0][j]], [x1[0][j],y1[0][j]], [x4[0][j],y4[0][j]], [x3[0][j],y3[0][j]]]) # clockwise
                                ztemp=[z2[0][j], z1[0][j], z4[0][j], z3[0][j]]
                            else :
                                pts1=np.float32([[x3[0][j],y3[0][j]], [x1[0][j],y1[0][j]], [x4[0][j],y4[0][j]], [x2[0][j],y2[0][j]]])
                                ztemp=[z3[0][j], z1[0][j], z4[0][j], z2[0][j]]
                    #print(pts1)
                    
                    mtrx=cv2.getPerspectiveTransform(pts1,pts2)
                    virtual_vec=np.float32([[xv[j]], [yv[j]], [1]])
                    r1=np.dot(mtrx,virtual_vec)
                    r1=r1/r1[2][0]
                    xint=r1[0][0].item() #x interpolation
                    yint=r1[1][0].item()
                    #print(r1)
                    if xint >=0 and xint <=1 and yint >=0 and yint<=1 :
                    #A,D interpol                    
                        a_=ztemp[0]*yint+ztemp[3]*(1-yint)
                    #B,C interpol
                        b_=ztemp[1]*yint+ztemp[2]*(1-yint)
                    #A' B' interpol
                        z.append(a_*(1-xint)+b_*xint)
                    else :
                        z.append((z1[0][j]+z2[0][j])/2)

            if cross%2 == 0 :
                #dist1=math.sqrt((x1[0][j]-xv[j])**2 + (y1[0][j]-yv[j])**2)
                #dist2=math.sqrt((x2[0][j]-xv[j])**2 + (y2[0][j]-yv[j])**2)
                #if dist1+dist2==0 :
                #    z.append(z1[0][j])
                #else :
                #    lerp=(dist1*z2[0][j]+dist2*z1[0][j])/(dist1+dist2)
                #    z.append(lerp)
                #z.append(z1[0][j])
                z.append((z1[0][j]+z2[0][j])/2)
                #del_inx.append(j)
          

        #remove index out of range
        #per_camera_virtual_indices=per_camera_indices.tolist() # my code to delete index
        #ktemp=0 # for count
        #for k in del_inx:
        #    k=k-ktemp
        #    del xv[k]
        #    del yv[k]
            #del per_camera_virtual_indices[k]
        #    ktemp +=1

        #per_camera_virtual_indices=torch.tensor(per_camera_virtual_indices, dtype=torch.long, device='cuda:0')
        #xv=torch.tensor(xv, dtype=torch.float32, device='cuda:0')
        #yv=torch.tensor(yv, dtype=torch.float32, device='cuda:0')
        #xv=xv.unsqueeze(0);yv=yv.unsqueeze(0)
        #xv=xv.view(-1,1); yv=yv.view(-1,1)

        del x1; del x2; del x3; del x4; del y1; del y2; del y3; del y4 #memory release
        del z1; del z2; del z3; del z4
        del xv; del yv
        del per_camera_indices2; del per_camera_indices3; del per_camera_indices4

        #print(xv)
        #print(torch.cat([xv,yv],dim=1))
        #print(per_camera_indices)
        #print(per_camera_virtual_indices)
        #print(per_camera_indices4.size())
        #print(point_labels)
        #per_camera_virtual_points[:,0]=xv
    
        z=torch.tensor(z, dtype=torch.float32, device='cuda:0')
        z=z.unsqueeze(0)
        #z=torch.as_tensor(z, dtype=torch.float32, device='cuda:0')
        #print(z)

        #r1=points.reshape(-1, 5)[per_camera_indices, :2] # real point x1, y1
        #r2=points.reshape(-1, 5)[per_camera_indices2, :2] # real point x2, y2
        #r3=points.reshape(-1, 5)[per_camera_indices3, :2] # real point x3, y3

        #a=y1*(z2-z3)+y2*(z3-z1)+y3*(z1-z2)
        #b=z1*(x2-x3)+z2*(x3-x1)+z3*(x1-x2)
        #c=x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)
        #d=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1)
        
        #diff1=torch.norm(vp-r1, dim=-1)
        #diff2=torch.norm(vp-r2, dim=-1)
        
        #diff2=math.sqrt(math.pow(xv-x2, 2) + math.pow(yv-y2, 2))
        
        #near1=torch.tensor([x1, y1])
        #near2=torch.tensor([x2, y2])
        #m1=torch.norm(near1-near2, dim=-1)
    
        #per_camera_virtual_points_depth = points.reshape(-1, 5)[per_camera_indices, 2].reshape(1, -1) # depth caculate only one point
        #per_camera_virtual_points_depth = (z1+z2)/2
        #per_camera_virtual_points_depth = (diff2*z1+diff1*z2)/(diff1+diff2)
        #per_camera_virtual_points_depth = -(a*xv+b*yv+d)/c
        per_camera_virtual_points_depth = z

        per_camera_virtual_points = per_camera_virtual_points[:, :2] # remove instance id, original
        #per_camera_virtual_points = torch.cat([xv,yv],dim=1) # my code
        per_camera_virtual_points_padded = torch.cat(
                [per_camera_virtual_points.transpose(1, 0).float(), 
                torch.ones((1, len(per_camera_virtual_points)), device=per_camera_indices.device, dtype=torch.float32)],
                dim=0
            )
        per_camera_virtual_points_3d = reverse_view_points(per_camera_virtual_points_padded, per_camera_virtual_points_depth, intrinsics[i])

        per_camera_virtual_points_3d[:3] = torch.matmul(torch.inverse(transforms[i]),
                torch.cat([
                        per_camera_virtual_points_3d[:3, :], 
                        torch.ones(1, per_camera_virtual_points_3d.shape[1], dtype=torch.float32, device=per_camera_indices.device)
                    ], dim=0)
            )[:3]

        all_virtual_points.append(per_camera_virtual_points_3d.transpose(1, 0))
        all_real_points.append(raw_points.reshape(1, -1, 4).repeat(num_camera, 1, 1).reshape(-1,4)[per_camera_indices][:, :3])
        all_point_labels.append(point_labels[per_camera_indices]) #list
        #all_virtual_labels.append(point_labels[per_camera_virtual_indices]) # my code

    all_virtual_points = torch.cat(all_virtual_points, dim=0)
    all_real_points = torch.cat(all_real_points, dim=0)
    all_point_labels = torch.cat(all_point_labels, dim=0)
    #all_virtual_labels = torch.cat(all_virtual_labels, dim=0)

    all_virtual_points = torch.cat([all_virtual_points, all_point_labels], dim=1) # my code, all point labels->all_virtual_labels

    real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)
    real_point_labels  = torch.max(real_point_labels, dim=0)[0]

    all_real_points = torch.cat([raw_points[foreground_real_point_mask.bool()], real_point_labels[foreground_real_point_mask.bool()]], dim=1)

    return all_virtual_points, all_real_points, foreground_real_point_mask.bool().nonzero(as_tuple=False).reshape(-1)

def init_detector(args):
    from CenterNet2.train_net import setup
    from detectron2.engine import DefaultPredictor

    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    return predictor 

def postprocess(res):
    result = res['instances']
    labels = result.pred_classes
    scores = result.scores 
    masks = result.pred_masks.reshape(scores.shape[0], 1600*900) 
    boxes = result.pred_boxes.tensor

    # remove empty mask and their scores / labels 
    empty_mask = masks.sum(dim=1) == 0

    labels = labels[~empty_mask] # no condition of empty mask
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]
    boxes = boxes[~empty_mask]
    masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
    return labels, scores, masks


@torch.no_grad()
def process_one_frame(info, predictor, data, num_camera=6): #info=data[0]
    all_cams_from_lidar = info['all_cams_from_lidar']
    all_cams_intrinsic = info['all_cams_intrinsic']
    lidar_points = read_file(info['lidar_path'])


    one_hot_labels = [] # declare empty array
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1 # [1][1]...[1] #10
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0) 

    masks = [] 
    labels = [] 
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1)

    result = predictor.model(data[1:]) # not include data[0]

    for camera_id in range(num_camera):
        pred_label, score, pred_mask = postprocess(result[camera_id])
        camera_id = torch.tensor(camera_id, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(pred_mask.shape[0], 1)
        pred_mask = torch.cat([pred_mask, camera_id], dim=1) # merge
        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

        masks.append(pred_mask)
        labels.append(transformed_labels)
    
    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)

    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic))
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, camera_ids], dim=-1)

    if len(masks) == 0:
        res = None
    else:
        res  = add_virtual_mask(masks, labels, P, to_tensor(lidar_points), 
            intrinsics=to_batch_tensor(all_cams_intrinsic), transforms=to_batch_tensor(all_cams_from_lidar))
    
    if res is not None:
        virtual_points, foreground_real_points, foreground_indices = res 
        return virtual_points.cpu().numpy(), foreground_real_points.cpu().numpy(), foreground_indices.cpu().numpy()
    else:
        return None 


def simple_collate(batch_list):
    assert len(batch_list)==1 # confirm it is true, batch_list's length is 1
    batch_list = batch_list[0]
    return batch_list


def main(args):
    predictor = init_detector(args)
    data_loader = DataLoader(
        PaintDataSet(args.info_path, predictor), #class declare
        batch_size=1,
        num_workers=6, #cpu core allocate
        collate_fn=simple_collate,
        pin_memory=False,
        shuffle=False
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)): #enum->tuple ('index', 'value')
        if len(data) == 0:
            continue 
        info = data[0]
        tokens = info['lidar_path'].split('/')
        output_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        
        res = process_one_frame(info, predictor, data)

        if res is not None:
            virtual_points, real_points, indices = res 
        else:
            virtual_points = np.zeros([0, 14]) #[]
            real_points = np.zeros([0, 15]) #[]
            indices = np.zeros(0) #0

        data_dict = {
            'virtual_points': virtual_points, 
            'real_points': real_points,
            'real_points_indice': indices
        }

        np.save(output_path, data_dict)
        # torch.cuda.empty_cache() if you get OOM error 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--info_path', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='c2_config/nuImages_CenterNet2_DLA_640_8x.yaml')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if not os.path.isdir('data/nuScenes/samples/LIDAR_TOP_VIRTUAL'):
        os.mkdir('data/nuScenes/samples/LIDAR_TOP_VIRTUAL')

    if not os.path.isdir('data/nuScenes/sweeps/LIDAR_TOP_VIRTUAL'):
        os.mkdir('data/nuScenes/sweeps/LIDAR_TOP_VIRTUAL')

    main(args)
