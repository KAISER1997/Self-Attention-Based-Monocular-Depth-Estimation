
# coding: utf-8

# In[13]:


from __future__ import division
import torch
import torch.nn.functional as F


import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from monodepth_utils import make_tensor
from monodepth_utils import tile


# In[14]:



def upsample(x,k):
    """Upsample input tensor by a factor of k
    """
    return F.interpolate(x, scale_factor=k, mode="nearest")



def check_sizes(input, input_name, expected):
   condition = [input.ndimension() == len(expected)]
   for i,size in enumerate(expected):
       if size.isdigit():
           condition.append(input.size(i) == int(size))
   assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))





def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat



def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat





def scale_pyramid(img, num_scales):
   """Scale the images into a dyadic pyramid
   :param img: Batch of images with shape [B, C, H, W].
   :param num_scales: (int) -> 4
   :return: img: the scaled image of the input [B, C, H, W].
   """
   scaled_imgs = [img]
   s = img.size()
   h = s[2]
   w = s[3]
   for i in range(num_scales - 1):
       ratio = 2 ** (i + 1)
       nh = h // ratio
       nw = w // ratio
       scaled_imgs.append(F.interpolate(img, size=[nh, nw], mode='bilinear', align_corners=True))
   return scaled_imgs


# In[21]:









def get_multi_scale_undistort_maps( camera_matrix, dist_coeffs, rotation_rect, new_camera_matrix):
    """Computes the joint undistortion and rectification transformation
    The undistorted image looks like original, as if it is captured with a camera using the camera
    matrix =new_camera_matrix and zero distortion
    https://docs.opencv.org/4.0.1/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    :param camera_matrix:
    :param dist_coeffs:
    :param rotation_rect:
    :param new_camera_matrix:
    :return:
    """
    scaled_imgs = [(512, 256), (256, 128), (128, 64), (64, 32)]
    map_x, map_y = zip(*[cv2.initUndistortRectifyMap(camera_matrix[i], dist_coeffs, rotation_rect,
                                                     new_camera_matrix[i], scaled_imgs[i],
                                                     cv2.CV_32FC1) for i in range(self.n)])
    return map_x, map_y

def undistortpoints( mesh_grid, camera_matrix, dist_coeffs, rotation_rect, new_camera_matrix):
    """Undistort grid
    https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistortpoints
    :param mesh_grid:
    :param camera_matrix:
    :param dist_coeffs:
    :param rotation_rect:
    :param new_camera_matrix:
    :return: undistorted grid [H * W x 1 X 2]
    """
    undistorted_points = cv2.undistortPoints(mesh_grid, camera_matrix, dist_coeffs,
                                             rotation_rect, new_camera_matrix)
    return undistorted_points

 
def scale_intrinsic(mat: np.array, scale ) -> np.array:
    """Scales the intrinsics based on the resized images for dyadic pyramid
    Refer Multiple View Geometry Richard Hartley Pg.156 Section:CCD Cameras
    :param mat: (torch.Tensor) -> the intrinsic matrix to scale
    :param scale: (Dict) -> { x, y } scaling parameters
    :return:(torch.Tensor): mat -> the scaled matrix
    """
    sx, sy = scale['x'], scale['y']
    mat[0, 0] *= sx
    mat[0, 2] *= sx
    mat[1, 1] *= sy
    mat[1, 2] *= sy
    return mat

def get_multi_scale_intrinsics(  intrinsic):
    """Returns multiple intrinsic matrices for different scales.
    :param intrinsic: the intrinsic parameters
    :return: numpy array of intrinsic params -- [num_scales x 3 x 3]
    """

    def make_intrinsics_matrix(fx, fy, cx, cy):
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0., 0., 1.]])

    intrinsics_multi_scale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(self.n):
        fx = intrinsic[0, 0] / (2 ** s)
        fy = intrinsic[1, 1] / (2 ** s)
        cx = intrinsic[0, 2] / (2 ** s)
        cy = intrinsic[1, 2] / (2 ** s)
        intrinsics_multi_scale.append(make_intrinsics_matrix(fx, fy, cx, cy))
    return np.stack(intrinsics_multi_scale, 0)  # num_scales x 3 x 3


def meshgrid(height: int, width: int) -> np.array:
    """Meshgrid in the absolute coordinates.
    :param height: image height
    :param width: image width
    :return: (np.array): mesh grid for the projections
    """
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    mesh_x, mesh_y = np.meshgrid(x, y)
    mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)
    grid = np.expand_dims(np.concatenate((mesh_x, mesh_y), axis=1), axis=1)  # [HW X 1 x 2]
    return grid


def img2cam( depth, pixel_coords, camera_matrix, distort_coeffs, new_camera_matrix,
            new_camera_matrix_inv, rotation_rect, device):
    """Transform coordinates in the pixel frame to the camera frame.
    :param depth: depth values for the pixels -- [B x 1 x H X W]
    :param pixel_coords: Pixel coordinates -- [H * W B x 1] (meshgrid)
    :param camera_matrix: Original distorted intrinsics matrix -- [3 x 3]
    :param distort_coeffs : Distortion coefficients k1, k2, p1, p2, k3
    :param new_camera_matrix_inv: Inverse scaled optimal intrinsics -- [3 x 3]
    :param new_camera_matrix: Scaled optimal intrinsics for undistorted points -- [3 x 3]
    :param device: cpu or gpu
    :return: camera_coords: Camera based coordinates -- [B x 3 x H * W]
    """
    camera_matrix = camera_matrix.cpu().detach().numpy()
    new_camera_matrix = new_camera_matrix.cpu().detach().numpy()
    batch,_,z1,z2 = depth.shape
#     print(z1,z2)
    hw=z1*z2
    depth = depth.reshape(batch, 1, -1)  # [B x 1 x H * W]
    rectifying_rotation = rotation_rect.cpu().detach().numpy()

    undistorted_cam_coords = undistortpoints(pixel_coords, camera_matrix, distort_coeffs, rectifying_rotation,
                                                  new_camera_matrix)
    undistorted_cam_coords = undistorted_cam_coords.transpose(1, 2, 0)  # [B x 2 x H * W]
    undistorted_cam_coords = make_tensor(undistorted_cam_coords, device)
    undistorted_cam_coords = tile(undistorted_cam_coords, 0, batch, device)
#     print(undistorted_cam_coords.shape)
#     print(hw)
    undistorted_cam_coords = torch.cat([undistorted_cam_coords, torch.ones(batch, 1, hw).to(device)], 1)
    
    # Normalize cam_coords as optimal new_camera_matrix was used
    undistorted_cam_coords_normalized = new_camera_matrix_inv @ undistorted_cam_coords
    return undistorted_cam_coords_normalized * depth  # [B x 3 x H * W]


def cam2world(undistorted_cam_coords: torch.Tensor, essential_mat: torch.Tensor, device: str) -> torch.Tensor:
    #here we will use our pose matrix to move from one scene to another
    """Convert homogeneous camera coordinates to world coordinates
    :param undistorted_cam_coords: Camera-based coordinates (transformed from image2cam) -- [B x 3 x H * W]
    :param essential_mat: The camera transform matrix -- [4 x 4]
    :param device: cpu or gpu
    :return: world_coords: World coordinate matrix -- [B x 4 x H * W]
    """
    batch, _, hw = undistorted_cam_coords.shape
    # [B x 4 x H * W]
    undistorted_cam_coords = torch.cat([undistorted_cam_coords, torch.ones(batch, 1, hw).to(device)], 1)
    # NOTE: Transformation with new rectified rotation with translation takes place
    world_coord = essential_mat @ undistorted_cam_coords
    return world_coord

 
def world2cam(world_coords: torch.Tensor, essential_identity: torch.Tensor) -> torch.Tensor:
    """Transform coordinates in the world to the camera frame.
    :param world_coords: First camera's projected world coordinates -- [B x 4 x H * W]
    :param essential_identity: Essential matrix -- [4 x 4]
    :return: cam_coords: Camera coordinates from the *other* camera's perspective -- [B x 4 x H * W]
    """
    cam_coords = essential_identity @ world_coords
    return cam_coords


    # TODO: Verify the logic
def cam2img( cam_coords: torch.Tensor, undistorted_intrinsic: torch.Tensor, height: int, width: int) -> tuple:
    """Transform coordinates in the camera frame to the pixel frame.
    Implements principled mask as explained in https://arxiv.org/pdf/1802.05522.pdf
    :param cam_coords: The camera based coords -- [B x 3 x H * W]
    :param undistorted_intrinsic: Camera intrinsics of undistorted image -- [3 x 3]
    :param height: image height
    :param width: image width
    :return: pixel_coords: The pixel coordinates corresponding to points -- [B x 2 x H * W]
    """
    batch, _, _ = cam_coords.shape
    pcoords = undistorted_intrinsic @ cam_coords
    x, y, z = [pcoords[:, i, :].unsqueeze(1) for i in range(3)]
    # avoid division by zero depth value
    z = z.clamp(min=1e-3)
    x_norm = 2 * (x / z) / (width - 1) - 1
    y_norm = 2 * (y / z) / (height - 1) - 1
    
    pcoords_norm = torch.cat([x_norm, y_norm], 1)  # b x 2 x hw
    # Principled mask
    x_mask = ((x_norm > 1) + (x_norm < -1))
    y_mask = ((y_norm > 1) + (y_norm < -1))
    mask = (x_mask | y_mask).reshape(batch, 1, height, width).float()
    return pcoords_norm, 1 - mask
 
def create_undistorted_images(src_image: torch.Tensor, undistorted_grid, device: str) -> torch.Tensor:
    """Creates undistorted images from distorted input images
    Transforms the source image using the specified map: dst (x,y) = src(map_x(x,y),map_y(x,y))
    :param src_image: Batch of distorted images with shape [B x 3 x H x W]
    :param undistorted_grid: Tensor of normalized x, y coordinates in [-1, 1], with shape [B x 2 x H x W]
    :return: Sampled image with shape [B, C, H, W]
    :param device: cpu or gpu
    :return: torch.Tensor of undistorted images [B x 3 x H x W]
    """
    batch_size, channels, height, width = src_image.shape
    undistorted_grid = undistorted_grid.transpose(1, 2, 0)  # [B x 2 x H x W]
    undistorted_grid = make_tensor(undistorted_grid, device)
    undistorted_grid[:,0,:]=2*undistorted_grid[:,0,:]/(width- 1) - 1
    undistorted_grid[:,1,:]=2*undistorted_grid[:,1,:]/(height- 1) - 1
    
    undistorted_grid = tile(undistorted_grid, 0, batch_size, device)

    flow_field = undistorted_grid.permute(0, 2, 1).reshape(batch_size, height, width, 2)
    output = F.grid_sample(src_image, flow_field, mode='bilinear', padding_mode='zeros')
    debug = False
    if debug:
        for i in range(output.size(0)):
            torchvision.utils.save_image(output[i, :, :, :], f'undistorted_img_{i}.png')
    return output



def bilinear_sampler(im: torch.Tensor, flow_field: torch.Tensor) -> torch.Tensor:
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    flow_field is the tensor specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    :param im: Batch of images with shape [B x 3 x H x W]
    :param flow_field: Tensor of normalized x, y coordinates in [-1, 1], with shape [B x 2 x H x W]
    :return: Sampled image with shape [B x 3 x H x W]
    """
    batch_size, channels, height, width = im.shape
    flow_field = flow_field.permute(0, 2, 1).reshape(batch_size, height, width, 2)
    output = F.grid_sample(im, flow_field, mode='bilinear', padding_mode='zeros')
    return output

 
def distort(cam_coords: torch.Tensor, distort_coeffs: torch.Tensor) -> torch.Tensor:
    """Distort pixel coordinates
    Implements OpenCV distortion model and adds the distortion to the image
    https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    :param cam_coords: The un-distorted camera coords -- [B x 3 x H * W]
    :param distort_coeffs : Distortion coefficients k1, k2, p1, p2, k3
    :return: pixel_coords: Undistort pixel coordinates corresponding to points -- [B x 3 x H * W]
    """
    x, y, z = [cam_coords[:, i, :].unsqueeze(1) for i in range(3)]
    # avoid division by zero depth value
    z = z.clamp(min=1e-3)
    x_ = x / z
    y_ = y / z
    r2 = x_ * x_ + y_ * y_
    # Radial distortion
    x_radial = x_ * (1 + distort_coeffs[0] * r2 + distort_coeffs[1] * r2 * r2 + distort_coeffs[4] * r2 * r2 * r2)
    y_radial = y_ * (1 + distort_coeffs[0] * r2 + distort_coeffs[1] * r2 * r2 + distort_coeffs[4] * r2 * r2 * r2)
    # Tangential distortion
    x_distorted = x_radial + (2 * distort_coeffs[2] * x_ * y_ + distort_coeffs[3] * (r2 + 2 * x_ * x_))
    y_distorted = y_radial + (distort_coeffs[2] * (r2 + 2 * y_ * y_) + 2 * distort_coeffs[3] * x_ * y_)
    ones = torch.ones(x_distorted.shape).type_as(x_distorted)
    cam_coords_distorted = torch.cat([x_distorted, y_distorted, ones], 1)
    return cam_coords_distorted


# In[22]:


def scale_intrinsic(mat: np.array, scale) -> np.array:
    """Scales the intrinsics based on the resized images for dyadic pyramid
    Refer Multiple View Geometry Richard Hartley Pg.156 Section:CCD Cameras
    :param mat: (torch.Tensor) -> the intrinsic matrix to scale
    :param scale: (Dict) -> { x, y } scaling parameters
    :return:(torch.Tensor): mat -> the scaled matrix
    """
    sx, sy = scale['x'], scale['y']
    mat[0, 0] *= sx
    mat[0, 2] *= sx
    mat[1, 1] *= sy
    mat[1, 2] *= sy
    return mat


# In[23]:


def inverse_warp_kaiser(src_img,depth,pose,distort,rotation_rect,camera_matrix,new_camera_matrix,device,flag_inv):
    #src_img=B*3*H*W tensor
#     camera_matrix=tensor (3*3)
    #depth=B*(H*W) tensor
    #distort= np array (5)
    
    #rotation_rect=3*3 tensor
    #pose=B*6 tensor
    batch,_,height,width=src_img.shape
    pose3_4=pose_vec2mat(pose).to(device)
    batch,_,_=pose3_4.shape
    s=torch.tensor([[[0,0,0,1]]],dtype=torch.float) 
    extra=tile(s.to(device),0,batch,device).to(device)
    if flag_inv==1:
        a=torch.zeros(4,3,4).to(device)
        eye=torch.zeros(4,3,4).to(device)
        eye[:,0,0]=eye[:,1,1]=eye[:,2,2]=1
        a[:,:,3]=1   #translation
        b=1-a  #rotation
        eye=eye.to(device).float()
        pose_left=torch.cat((pose3_4*b.float(),extra),1).to(device)  #rotation
        pose_right=torch.cat((pose3_4*a.float()+eye,extra),1).to(device) #trans
        pose4_4=torch.matmul(pose_left,pose_right)
    else:    
        pose4_4=torch.cat((pose3_4,extra),1).to(device)
    gridz=meshgrid(height,width)
    new_camera_matrix_inv=torch.inverse(new_camera_matrix).to(device)
    cam_coordinate=img2cam( depth, gridz, camera_matrix, distort , new_camera_matrix,new_camera_matrix_inv, rotation_rect,device)
    after_shifitng_to_next_frame=cam2world(cam_coordinate , pose4_4 , device)
    other_cam_coords = after_shifitng_to_next_frame[:, :3, :]

#OBSRVE THIS PART CAREFULLY
    new_img_coordinates,mask=cam2img(other_cam_coords, new_camera_matrix,  height,width)

    torch.backends.cudnn.enabled = False
    #observe here
    transformed_image=bilinear_sampler(src_img, new_img_coordinates.float())
#     print("f6-"+str(transformed_image.requires_grad))
    torch.backends.cudnn.enabled =True
    return(transformed_image,mask)


    


# In[24]:


def ungrid_und_img(img,camera_matrix,new_camera_matrix,distort,rotation_rect,device):
    #used to undistort a batch of image
##TO BE CHECKED
    #img- B X 3 X H X W
    batch,_,height,width=img.shape
    
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distort, rotation_rect,
                                                     new_camera_matrix, (width,height),
                                                     cv2.CV_32FC1) 
    
    map_x=2*map_x.reshape(1,height,width,1)/(width-1) -1
    map_y=2*map_y.reshape(1,height,width,1)/(height-1) -1

    grid=torch.cat([torch.tensor(map_x),torch.tensor(map_y)],3)
    grid_batch=tile(grid.to(device), 0, batch, device)

    torch.backends.cudnn.enabled = False
    outputz = F.grid_sample(img, grid_batch.to(device), mode='bilinear')
    
    torch.backends.cudnn.enabled = True
    
    return(outputz)


# In[25]:


#height=256 width=512

def inverse_warp_pyramid(ref_Img,pyramid_depth,ht,wid,LorR,poze,device,config_pathz,flag_inv):
    ref_img_pyramid=scale_pyramid(ref_Img,4) #distorted images
    
    warped_pyramid=[]
    mask_pyramid=[]
    for i in range(4):
        dpf=1/pyramid_depth[i]
        Izcamera_matrix,Iznew_camera_matrix,Izdistort,Izrotation_rect=get_other_parameters(config_pathz,LorR,[wid,ht])
#         dpf=ungrid_und_img(dpf,Izcamera_matrix.cpu().numpy(),Iznew_camera_matrix.cpu().numpy(),Izdistort,Izrotation_rect.cpu().numpy()) #undistorting the disparity
        dp=dpf
        undistorted_ref=ungrid_und_img(ref_img_pyramid[i],Izcamera_matrix.cpu().numpy(),Iznew_camera_matrix.cpu().numpy(),Izdistort,Izrotation_rect.cpu().numpy(),device)

        warped_ref,mask=inverse_warp_kaiser(undistorted_ref.to(device),dp,poze,Izdistort,Izrotation_rect.to(device),Izcamera_matrix.to(device),Iznew_camera_matrix.to(device),device,flag_inv)
        warped_pyramid.append(warped_ref)
        mask_pyramid.append(mask)
        wid=wid//2
        ht=ht//2
        if i==3:
            break
        
        
    return(warped_pyramid,mask_pyramid)


# In[26]:

def inverse_warp_pyramid2(ref_Img,pyramid_depth,ht,wid,LorR,poze,device,config_pathz,flag_inv):
    ref_img_pyramid=scale_pyramid(ref_Img,4) #distorted images
    
    warped_pyramid=[]
    mask_pyramid=[]
    for i in range(4):
        dpf=1/upsample(pyramid_depth[i],2**i)
        Izcamera_matrix,Iznew_camera_matrix,Izdistort,Izrotation_rect=get_other_parameters(config_pathz,LorR,[wid,ht])
#         dpf=ungrid_und_img(dpf,Izcamera_matrix.cpu().numpy(),Iznew_camera_matrix.cpu().numpy(),Izdistort,Izrotation_rect.cpu().numpy()) #undistorting the disparity
        dp=dpf
        undistorted_ref=ungrid_und_img(ref_Img,Izcamera_matrix.cpu().numpy(),Iznew_camera_matrix.cpu().numpy(),Izdistort,Izrotation_rect.cpu().numpy(),device)

        warped_ref,mask=inverse_warp_kaiser(undistorted_ref.to(device),dp,poze,Izdistort,Izrotation_rect.to(device),Izcamera_matrix.to(device),Iznew_camera_matrix.to(device),device,flag_inv)
        warped_pyramid.append(warped_ref)
        mask_pyramid.append(mask)
#         wid=wid//2
#         ht=ht//2
        if i==3:
            break
        
        
    return(warped_pyramid,mask_pyramid)


def inverse_warp_pyramid3(ref_Img,pyramid_depth,ht,wid,LorR,poze,device,config_pathz,flag_inv):
    ref_img_pyramid=scale_pyramid(ref_Img,4) #distorted images
    
    warped_pyramid=[]
    mask_pyramid=[]
    for i in range(4):
        dpf=1/pyramid_depth[i]
        Izcamera_matrix,Iznew_camera_matrix,Izdistort,Izrotation_rect=get_other_parameters(config_pathz,LorR,[wid,ht])
#         dpf=ungrid_und_img(dpf,Izcamera_matrix.cpu().numpy(),Iznew_camera_matrix.cpu().numpy(),Izdistort,Izrotation_rect.cpu().numpy()) #undistorting the disparity
        dp=dpf
        undistorted_ref=ungrid_und_img(ref_Img,Izcamera_matrix.cpu().numpy(),Iznew_camera_matrix.cpu().numpy(),Izdistort,Izrotation_rect.cpu().numpy(),device)

        warped_ref,mask=inverse_warp_kaiser(undistorted_ref.to(device),dp,poze,Izdistort,Izrotation_rect.to(device),Izcamera_matrix.to(device),Iznew_camera_matrix.to(device),device,flag_inv)
        warped_pyramid.append(warped_ref)
        mask_pyramid.append(mask)
#         wid=wid//2
#         ht=ht//2
        if i==3:
            break
        
        
    return(warped_pyramid,mask_pyramid)














def get_other_parameters(config_path,LorR,size:tuple):
    #:param size: (width, height)
    cam_params = json.load(open(config_path))
    height = cam_params['height']
    width = cam_params['width']
    scale = {'y': float(size[1]) / height,
             'x': float(size[0]) / width}
    
    left_intrinsic = np.array(cam_params['intrinsics']['left_camera'], dtype=np.float32).reshape(3, 3)
    right_intrinsic = np.array(cam_params['intrinsics']['right_camera'], dtype=np.float32).reshape(3, 3)



    distort_left = np.array(cam_params['distort_left'], dtype=np.float32).reshape(5)
    distort_right = np.array(cam_params['distort_right'], dtype=np.float32).reshape(5)

    left_R_rect_02 = np.array(cam_params['R_rect_02'], dtype=np.float32).reshape(3, 3)
    right_R_rect_03 = np.array(cam_params['R_rect_03'], dtype=np.float32).reshape(3, 3)

    left_new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(left_intrinsic,
                                                               distort_left,
                                                               (width, height), 1)

    right_new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(right_intrinsic,
                                                                distort_right,
                                                                (width, height), 1)
    left_intrinsic_scaled = scale_intrinsic(left_intrinsic, scale)
    left_optimal_intrinsic =scale_intrinsic(left_new_intrinsic, scale)
    right_intrinsic_scaled =scale_intrinsic(right_intrinsic, scale)
    right_optimal_intrinsic =scale_intrinsic(right_new_intrinsic, scale)
    if LorR=='2':
        return torch.tensor(left_intrinsic_scaled),torch.tensor(left_optimal_intrinsic),distort_left.reshape(5),torch.tensor(left_R_rect_02)
    else:
        return torch.tensor(right_intrinsic_scaled),torch.tensor(right_optimal_intrinsic),distort_right.reshape(5),torch.tensor(right_R_rect_03)

