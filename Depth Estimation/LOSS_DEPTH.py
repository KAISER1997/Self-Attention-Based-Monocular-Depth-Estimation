
# coding: utf-8

# In[2]:


import torch.nn as nn
import torch.nn.functional as F
import torch

# In[3]:
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

class SSIM_CLASS_mono2(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_CLASS_mono2, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
def SSIM_CLASS(x, y):
    c_1 = 0.01 ** 2
    c_2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + c_1) * (2 * sigma_xy + c_2)
    SSIM_d = (mu_x_sq + mu_y_sq + c_1) * (sigma_x + sigma_y + c_2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

# In[4]:

def SSIM(x, y):
    c_1 = 0.01 ** 2
    c_2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + c_1) * (2 * sigma_xy + c_2)
    SSIM_d = (mu_x_sq + mu_y_sq + c_1) * (sigma_x + sigma_y + c_2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

def loss(src,target):
    return(torch.mean(SSIM(src,target)))




def compute_reprojection_loss2( pred, target,mask):
    """Computes reprojection loss between a batch of predicted and target images
    """
    
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean()
    ssim_loss =loss(pred,target) #SSIM_CLASS(pred, target)*-F.max_pool2d(-mask, 3, 1)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss  

def compute_reprojection_loss3( pred, target,mask):
    """Computes reprojection loss between a batch of predicted and target images
    """
    
    abs_diff = torch.abs(target - pred)
    l1_loss = (abs_diff*mask).sum()/(mask.sum())
    ssim_loss = (SSIM_CLASS(pred, target)*-F.max_pool2d(-mask, 3, 1)).sum()/((-F.max_pool2d(-mask, 3, 1)).sum())
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss  








def compute_reprojection_loss( pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
#     ssim_mono2=SSIM()
#     abs_diff = torch.abs(target - pred)
#     l1_loss = abs_diff.mean(1, True)
#     ssim_loss = ssim_mono2(pred, target).mean(1, True)
#     reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return loss(pred,target)   

def compute_reprojection_loss_mono2( pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    ssim_mono2=SSIM_CLASS_mono2()
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim_loss = ssim_mono2(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss
# In[5]:


def gradient_x(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


def gradient_y(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy
def depth_smoothness( depth, pyramid):
    """Smoothens the output depth map
    :param depth: tensor depth
    :param pyramid: input tensor image comprising 4 scales
    :return: smoothened depth map pyramid list of images
    """
    depth_gradients_x = [gradient_x(d) for d in depth]
    depth_gradients_y = [gradient_y(d) for d in depth]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [depth_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [depth_gradients_y[i] * weights_y[i] for i in range(4)]

    return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(4)]
ccx=[]
def loss_pyramid(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
        reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],mk1_p[i]) 
        reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],mk2_p[i])
#         print(reproj_1.shape)
#         reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
#         identity_loss1=compute_reprojection_loss(ref1_pyramid[i],target_pyramid[i])
#         identity_loss2=compute_reprojection_loss(ref2_pyramid[i],target_pyramid[i])
#         identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#         concat=torch.cat([identity_losses,reprojection_losses],1)
        
# #         print(concat.shape)
# #         print(concat.shape)
#         to_op,idxa=torch.min(concat,1)
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
        
#         to_optimize=reprojection_losses  #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
        net_loss=net_loss+(reproj_1+reproj_2)
        
        
        if i==2:
            break
        
    return(net_loss/3,0)
        

    
def loss_pyramid2(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
        reproj_1=compute_reprojection_loss2(warped_pyramid1[i],target_pyramid[i],mk1_p[i]) 
        reproj_2=compute_reprojection_loss2(warped_pyramid2[i],target_pyramid[i],mk2_p[i])
#         print(reproj_1.shape)
#         reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
#         identity_loss1=compute_reprojection_loss(ref1_pyramid[i],target_pyramid[i])
#         identity_loss2=compute_reprojection_loss(ref2_pyramid[i],target_pyramid[i])
#         identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#         concat=torch.cat([identity_losses,reprojection_losses],1)
        
# #         print(concat.shape)
# #         print(concat.shape)
#         to_op,idxa=torch.min(concat,1)
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
        
        to_optimize=reproj_1+reproj_2  #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
        net_loss=net_loss+(to_optimize)
        
        
        if i==1:
            break
        
    return(net_loss,0)



def V15loss_pyramid2(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
#     dep_smooth=depth_smoothness(depth_list,target_pyramid)
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
        reproj_1=compute_reprojection_loss2(warped_pyramid1[i],target_pyramid[i],mk1_p[i]) 
        reproj_2=compute_reprojection_loss2(warped_pyramid2[i],target_pyramid[i],mk2_p[i])
#         print(reproj_1.shape)
#         reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
#         identity_loss1=compute_reprojection_loss(ref1_pyramid[i],target_pyramid[i])
#         identity_loss2=compute_reprojection_loss(ref2_pyramid[i],target_pyramid[i])
#         identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#         concat=torch.cat([identity_losses,reprojection_losses],1)
        
# #         print(concat.shape)
# #         print(concat.shape)
#         to_op,idxa=torch.min(concat,1)
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
        
        to_optimize=reproj_1+reproj_2  #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
        net_loss=net_loss+(to_optimize)
        
        
        if i==0:
            break
        
    return(net_loss,0)
            
    
def loss_pyramid4(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
#         identity_loss1=torch.abs(target_pyramid[i]-ref1_pyramid[i]).mean(1,keepdim=True)
#         identity_loss2=torch.abs(target_pyramid[i]-ref2_pyramid[i]).mean(1,keepdim=True)
#         abs_loss1=torch.abs(warped_pyramid1[i]-target_pyramid[i]).mean(1,keepdim=True)
#         abs_loss2=torch.abs(warped_pyramid2[i]-target_pyramid[i]).mean(1,keepdim=True)
#         concat1=torch.cat([identity_loss1,abs_loss1],1)
#         concat2=torch.cat([identity_loss2,abs_loss2],1)
#         _,idxa_1=torch.min(concat1,1,keepdim=True)
#         _,idxa_2=torch.min(concat2,1,keepdim=True)
#         id1_mask=idxa_1>0
#         id2_mask=idxa_2>0
                            
                            
#         vt.append(id1_mask)
#         vt.append(id2_mask)
#         reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],id1_mask.float().to(device)) 
#         reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],id2_mask.float().to(device))
# #         print(reproj_1.shape)
          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_pyramid[i])
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_pyramid[i])
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            
          identity_loss1=compute_reprojection_loss_mono2(ref1_pyramid[i],target_pyramid[i])
          identity_loss2=compute_reprojection_loss_mono2(ref2_pyramid[i],target_pyramid[i])
          identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)
          conc1=torch.cat([identity_loss1,reproj_1],1)
          conc2=torch.cat([identity_loss2,reproj_2],1)
# #         print(concat.shape)
# #         print(concat.shape)
          to_op1,idxa1=torch.min(conc1,1,keepdim=True)
          to_op2,idxa2=torch.min(conc2,1,keepdim=True)
          id_mask1=idxa1>0
          id_mask2=idxa2>0
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
          vt.append(idxa1>0)
          vt.append(idxa2>0) 
          loss1=to_op1.mean()
          loss2=to_op2.mean()
        
          to_optimize=loss1+loss2  #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==1:
                break
        
    return(net_loss,vt)    
def loss_pyramid_RAND(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
#         identity_loss1=torch.abs(target_pyramid[i]-ref1_pyramid[i]).mean(1,keepdim=True)
#         identity_loss2=torch.abs(target_pyramid[i]-ref2_pyramid[i]).mean(1,keepdim=True)
#         abs_loss1=torch.abs(warped_pyramid1[i]-target_pyramid[i]).mean(1,keepdim=True)
#         abs_loss2=torch.abs(warped_pyramid2[i]-target_pyramid[i]).mean(1,keepdim=True)
#         concat1=torch.cat([identity_loss1,abs_loss1],1)
#         concat2=torch.cat([identity_loss2,abs_loss2],1)
#         _,idxa_1=torch.min(concat1,1,keepdim=True)
#         _,idxa_2=torch.min(concat2,1,keepdim=True)
#         id1_mask=idxa_1>0
#         id2_mask=idxa_2>0
                            
                            
#         vt.append(id1_mask)
#         vt.append(id2_mask)
#         reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],id1_mask.float().to(device)) 
#         reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],id2_mask.float().to(device))
# #         print(reproj_1.shape)
          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_pyramid[i])
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_pyramid[i])
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            
          identity_loss1=compute_reprojection_loss_mono2(ref1_pyramid[i],target_pyramid[i])
          identity_loss2=compute_reprojection_loss_mono2(ref2_pyramid[i],target_pyramid[i])
          identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)
          conc1=torch.cat([identity_loss1,reproj_1],1)
          conc2=torch.cat([identity_loss2,reproj_2],1)
# #         print(concat.shape)
# #         print(concat.shape)
          to_op1,idxa1=torch.min(conc1,1,keepdim=True)
          to_op2,idxa2=torch.min(conc2,1,keepdim=True)
          a=reproj_1.shape
          id_mask1=torch.rand(a[0],a[1],a[2],a[3]).to(device)
          id_mask2=torch.rand(a[0],a[1],a[2],a[3]).to(device)
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
          vt.append(idxa1>0)
          vt.append(idxa2>0) 
          loss1=(reproj_1*id_mask1.float())#.sum(2).sum(2))#/(id_mask1.sum(2).sum(2).float())
          loss2=(reproj_2*id_mask2.float())#.sum(2).sum(2))#/(id_mask2.sum(2).sum(2).float())
        
          to_optimize=loss1.mean()+loss2.mean()  #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==1:
                break
        
    return(net_loss,vt)    
    
def loss_pyramid7(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
#         identity_loss1=torch.abs(target_pyramid[i]-ref1_pyramid[i]).mean(1,keepdim=True)
#         identity_loss2=torch.abs(target_pyramid[i]-ref2_pyramid[i]).mean(1,keepdim=True)
#         abs_loss1=torch.abs(warped_pyramid1[i]-target_pyramid[i]).mean(1,keepdim=True)
#         abs_loss2=torch.abs(warped_pyramid2[i]-target_pyramid[i]).mean(1,keepdim=True)
#         concat1=torch.cat([identity_loss1,abs_loss1],1)
#         concat2=torch.cat([identity_loss2,abs_loss2],1)
#         _,idxa_1=torch.min(concat1,1,keepdim=True)
#         _,idxa_2=torch.min(concat2,1,keepdim=True)
#         id1_mask=idxa_1>0
#         id2_mask=idxa_2>0
                            
                            
#         vt.append(id1_mask)
#         vt.append(id2_mask)
#         reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],id1_mask.float().to(device)) 
#         reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],id2_mask.float().to(device))
# #         print(reproj_1.shape)
          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_pyramid[i])
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_pyramid[i])
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            
          identity_loss1=compute_reprojection_loss_mono2(ref1_pyramid[i],target_pyramid[i])
          identity_loss2=compute_reprojection_loss_mono2(ref2_pyramid[i],target_pyramid[i])
          identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)
          conc1=torch.cat([identity_loss1,reproj_1],1)
          conc2=torch.cat([identity_loss2,reproj_2],1)
# #         print(concat.shape)
# #         print(concat.shape)
          to_op1,idxa1=torch.min(conc1,1,keepdim=True)
          to_op2,idxa2=torch.min(conc2,1,keepdim=True)
          id_mask1=idxa1>0
          id_mask2=idxa2>0
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
          vt.append(idxa1>0)
          vt.append(idxa2>0) 
          loss1=to_op1.mean()
          loss2=to_op2.mean()
          smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
          to_optimize=loss1+loss2+0.05*smooth_loss
          #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==3:
                break
        
    return(net_loss,vt)        
    
def loss_pyramid7_2(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
#         identity_loss1=torch.abs(target_pyramid[i]-ref1_pyramid[i]).mean(1,keepdim=True)
#         identity_loss2=torch.abs(target_pyramid[i]-ref2_pyramid[i]).mean(1,keepdim=True)
#         abs_loss1=torch.abs(warped_pyramid1[i]-target_pyramid[i]).mean(1,keepdim=True)
#         abs_loss2=torch.abs(warped_pyramid2[i]-target_pyramid[i]).mean(1,keepdim=True)
#         concat1=torch.cat([identity_loss1,abs_loss1],1)
#         concat2=torch.cat([identity_loss2,abs_loss2],1)
#         _,idxa_1=torch.min(concat1,1,keepdim=True)
#         _,idxa_2=torch.min(concat2,1,keepdim=True)
#         id1_mask=idxa_1>0
#         id2_mask=idxa_2>0
                            
                            
#         vt.append(id1_mask)
#         vt.append(id2_mask)
#         reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],id1_mask.float().to(device)) 
#         reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],id2_mask.float().to(device))
# #         print(reproj_1.shape)
          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_Img)
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_Img)
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            
          identity_loss1=compute_reprojection_loss_mono2(ref1,target_Img)
          identity_loss2=compute_reprojection_loss_mono2(ref2,target_Img)
          identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)
          conc1=torch.cat([identity_loss1,reproj_1],1)
          conc2=torch.cat([identity_loss2,reproj_2],1)
# #         print(concat.shape)
# #         print(concat.shape)
          to_op1,idxa1=torch.min(conc1,1,keepdim=True)
          to_op2,idxa2=torch.min(conc2,1,keepdim=True)
          id_mask1=idxa1>0
          id_mask2=idxa2>0
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
          vt.append(idxa1>0)
          vt.append(idxa2>0) 
          loss1=to_op1.mean()
          loss2=to_op2.mean()
          smooth_loss=torch.mean(torch.abs(dep_smooth[i]))
          to_optimize=loss1+loss2+0.05*smooth_loss
          #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==3:
                break
        
    return(net_loss,vt)     



def loss_pyramid7_3(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=[target_Img,target_Img,target_Img,target_Img] #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
#         identity_loss1=torch.abs(target_pyramid[i]-ref1_pyramid[i]).mean(1,keepdim=True)
#         identity_loss2=torch.abs(target_pyramid[i]-ref2_pyramid[i]).mean(1,keepdim=True)
#         abs_loss1=torch.abs(warped_pyramid1[i]-target_pyramid[i]).mean(1,keepdim=True)
#         abs_loss2=torch.abs(warped_pyramid2[i]-target_pyramid[i]).mean(1,keepdim=True)
#         concat1=torch.cat([identity_loss1,abs_loss1],1)
#         concat2=torch.cat([identity_loss2,abs_loss2],1)
#         _,idxa_1=torch.min(concat1,1,keepdim=True)
#         _,idxa_2=torch.min(concat2,1,keepdim=True)
#         id1_mask=idxa_1>0
#         id2_mask=idxa_2>0
                            
                            
#         vt.append(id1_mask)
#         vt.append(id2_mask)
#         reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],id1_mask.float().to(device)) 
#         reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],id2_mask.float().to(device))
# #         print(reproj_1.shape)
          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_Img)
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_Img)
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            
          identity_loss1=compute_reprojection_loss_mono2(ref1,target_Img)
          identity_loss2=compute_reprojection_loss_mono2(ref2,target_Img)
          identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)
          conc1=torch.cat([identity_loss1,reproj_1],1)
          conc2=torch.cat([identity_loss2,reproj_2],1)
# #         print(concat.shape)
# #         print(concat.shape)
          to_op1,idxa1=torch.min(conc1,1,keepdim=True)
          to_op2,idxa2=torch.min(conc2,1,keepdim=True)
          id_mask1=idxa1>0
          id_mask2=idxa2>0
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
          vt.append(idxa1>0)
          vt.append(idxa2>0) 
          loss1=to_op1.mean()
          loss2=to_op2.mean()
          smooth_loss=torch.mean(torch.abs(dep_smooth[i]))
          to_optimize=loss1+loss2+0.05*smooth_loss
          #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==3:
                break
        
    return(net_loss,vt)      



def loss_pyramid7_4(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        
#         print(target_pyramid[i].shape)
#         print(target_pyramid[i].shape,warped_pyramid1[i].shape)
#         identity_loss1=torch.abs(target_pyramid[i]-ref1_pyramid[i]).mean(1,keepdim=True)
#         identity_loss2=torch.abs(target_pyramid[i]-ref2_pyramid[i]).mean(1,keepdim=True)
#         abs_loss1=torch.abs(warped_pyramid1[i]-target_pyramid[i]).mean(1,keepdim=True)
#         abs_loss2=torch.abs(warped_pyramid2[i]-target_pyramid[i]).mean(1,keepdim=True)
#         concat1=torch.cat([identity_loss1,abs_loss1],1)
#         concat2=torch.cat([identity_loss2,abs_loss2],1)
#         _,idxa_1=torch.min(concat1,1,keepdim=True)
#         _,idxa_2=torch.min(concat2,1,keepdim=True)
#         id1_mask=idxa_1>0
#         id2_mask=idxa_2>0
                            
                            
#         vt.append(id1_mask)
#         vt.append(id2_mask)
#         reproj_1=compute_reprojection_loss3(warped_pyramid1[i],target_pyramid[i],id1_mask.float().to(device)) 
#         reproj_2=compute_reprojection_loss3(warped_pyramid2[i],target_pyramid[i],id2_mask.float().to(device))
# #         print(reproj_1.shape)
          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_Img)
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_Img)
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            
          identity_loss1=compute_reprojection_loss_mono2(ref1,target_Img)
          identity_loss2=compute_reprojection_loss_mono2(ref2,target_Img)
          identity_losses=torch.cat([identity_loss1,identity_loss2],1)#.mean(1,keepdim=True).to(device)
#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)
          conc1=torch.cat([identity_loss1,reproj_1],1)
          conc2=torch.cat([identity_loss2,reproj_2],1)
# #         print(concat.shape)
# #         print(concat.shape)
          to_op1,idxa1=torch.min(conc1,1,keepdim=True)
          to_op2,idxa2=torch.min(conc2,1,keepdim=True)
          id_mask1=idxa1>0
          id_mask2=idxa2>0
#         id_mask=idxa>1
#         print(concat.shape,id_mask.shape,torch.max(idxa),reprojection_losses.shape)
#         ccx.append(idxa)
#         to_optimize=reprojection_losses
#         id_=id_<1
          vt.append(idxa1>0)
          vt.append(idxa2>0) 
          loss1=to_op1.mean()
          loss2=to_op2.mean()
          smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/(2**i)
          to_optimize=loss1+loss2+0.05*smooth_loss
          #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==3:
                break
        
    return(net_loss,vt)        





def loss_pyramid7_nomask(warped_pyramid1,warped_pyramid2,target_Img,ref1,ref2,depth_list,mk1_p,mk2_p,device):
    net_loss=0
    target_pyramid=scale_pyramid(target_Img,4) #t
    ref1_pyramid=scale_pyramid(ref1,4) #t-1
    ref2_pyramid=scale_pyramid(ref2,4) #t+1
    prev=0
    dep_smooth=depth_smoothness(depth_list,target_pyramid)
    
    vt=[]
#     print(len(depth_list),len(target_pyramid))
    for i in range(4):
        

          reproj_1=compute_reprojection_loss_mono2(warped_pyramid1[i],target_Img)
          reproj_2=compute_reprojection_loss_mono2(warped_pyramid2[i],target_Img)
          reprojection_losses=torch.cat([reproj_1,reproj_2],1).mean(1,keepdim=True).to(device)
            


#         identity_losses=identity_losses+torch.randn(identity_losses.shape).to(device)*0.00001
#           concat=torch.cat([identity_losses,reprojection_losses],1)

# #         print(concat.shape)
# #         print(concat.shape)
          to_op1=reproj_1
          to_op2=reproj_2
          loss1=to_op1.mean()
          loss2=to_op2.mean()
          smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/(2**i)
          to_optimize=loss1+loss2+0.05*smooth_loss
          #*id_mask.unsqueeze(1).float()
#         smooth_loss=torch.mean(torch.abs(dep_smooth[i]))/ (2**i)
#         print(to_optimize.mean())
#         net_loss=net_loss+torch.sum(to_optimize)/((torch.sum(mk1_p[i])+torch.sum(mk2_p[i]))/2)+0.05*smooth_loss
#         print(to_optimize.mean())
#         print(l1_loss,ssim_loss,smooth_loss)
          net_loss=net_loss+to_optimize
        
        
          if i==3:
                break
        
    return(net_loss,0)      