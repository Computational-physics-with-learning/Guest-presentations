
import torch 
from torch.utils.data import Dataset
from scipy.io import loadmat
import os 
import numpy as np 

angular_range_to_idx = {
    90: 181,
    80: 161,
    70: 141,
    60: 121,
    50: 101,
    40: 81,
    30: 61
}


class ChallengeData(Dataset):
    """
    Load the 5 phantom provided by the challenge organizers
    """
    def __init__(self,  angular_range, random_start_angle = False, base_dir = "htc_data/", include_fbp=False):
        
        self.angular_range = angular_range
        self.random_start_angle = random_start_angle

        self.base_dir = base_dir

        # [sinogram, segmentation, fbp]
        self.files = [["htc2022_solid_disc_full.mat", "htc2022_solid_disc_full_recon_fbp_seg.mat", "htc2022_solid_disc_full_recon_fbp.mat"], 
                      ["htc2022_ta_full.mat", "htc2022_ta_full_recon_fbp_seg.mat", "htc2022_ta_full_recon_fbp.mat"], 
                      ["htc2022_tb_full.mat", "htc2022_tb_full_recon_fbp_seg.mat", "htc2022_tb_full_recon_fbp.mat"], 
                      ["htc2022_tc_full.mat", "htc2022_tc_full_recon_fbp_seg.mat", "htc2022_tc_full_recon_fbp.mat"], 
                      ["htc2022_td_full.mat", "htc2022_td_full_recon_fbp_seg.mat", "htc2022_td_full_recon_fbp.mat"]]

        self.include_fbp = include_fbp

        if self.random_start_angle:
            print("WARNING: random_start_angle in ChallengeData is set to TRUE. \n You have to define the RayTrafo and FBP to use [start_angle, start_angle + angular_range] (Default is [0, angular_range]).")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, IDX):

        disk_file = self.files[IDX]

        t_full = loadmat(os.path.join(self.base_dir, disk_file[0]), struct_as_record=False, simplify_cells=True)
        t_seg = loadmat(os.path.join(self.base_dir, disk_file[1]), struct_as_record=False, simplify_cells=True)

        if self.include_fbp:
            t_fbp = loadmat(os.path.join(self.base_dir, disk_file[2]), struct_as_record=False, simplify_cells=True)
            t_fbp = torch.from_numpy(t_fbp['reconFullFbp']).unsqueeze(0).float()

        if self.random_start_angle:
            start_angle = np.random.randint(low=0, high=721 - angular_range_to_idx[self.angular_range])
            stop_angle = start_angle + angular_range_to_idx[self.angular_range]

        else:
            start_angle = 0
            stop_angle = angular_range_to_idx[self.angular_range]

        t_sino = torch.from_numpy(t_full['CtDataFull']['sinogram'][start_angle:stop_angle, :]).unsqueeze(0).float()

        t_seg = torch.from_numpy(t_seg['reconFullFbpSeg']).unsqueeze(0).float()

        if self.include_fbp:

            return t_sino, t_seg, torch.from_numpy(np.asarray([start_angle, stop_angle])), t_fbp 
        else:
            return t_sino, t_seg, torch.from_numpy(np.asarray([start_angle, stop_angle]))



class FullChallengeData(Dataset):
    """
    Load the 5 phantom provided by the challenge organizers
    """
    def __init__(self, base_dir = "htc_data/", include_fbp=False):
        
        self.base_dir = base_dir

        # [sinogram, segmentation, fbp]
        self.files = [["htc2022_solid_disc_full.mat", "htc2022_solid_disc_full_recon_fbp_seg.mat", "htc2022_solid_disc_full_recon_fbp.mat"], 
                      ["htc2022_ta_full.mat", "htc2022_ta_full_recon_fbp_seg.mat", "htc2022_ta_full_recon_fbp.mat"], 
                      ["htc2022_tb_full.mat", "htc2022_tb_full_recon_fbp_seg.mat", "htc2022_tb_full_recon_fbp.mat"], 
                      ["htc2022_tc_full.mat", "htc2022_tc_full_recon_fbp_seg.mat", "htc2022_tc_full_recon_fbp.mat"], 
                      ["htc2022_td_full.mat", "htc2022_td_full_recon_fbp_seg.mat", "htc2022_td_full_recon_fbp.mat"]]

        self.include_fbp = include_fbp


    def __len__(self):
        return len(self.files)

    def __getitem__(self, IDX):

        disk_file = self.files[IDX]

        t_full = loadmat(os.path.join(self.base_dir, disk_file[0]), struct_as_record=False, simplify_cells=True)
        t_seg = loadmat(os.path.join(self.base_dir, disk_file[1]), struct_as_record=False, simplify_cells=True)

        if self.include_fbp:
            t_fbp = loadmat(os.path.join(self.base_dir, disk_file[2]), struct_as_record=False, simplify_cells=True)
            t_fbp = torch.from_numpy(t_fbp['reconFullFbp']).unsqueeze(0).float()

       
        t_sino = torch.from_numpy(t_full['CtDataFull']['sinogram']).unsqueeze(0).float() 

        t_seg = torch.from_numpy(t_seg['reconFullFbpSeg']).unsqueeze(0).float()

        if self.include_fbp:

            return t_sino, t_seg, t_fbp 
        else:
            return t_sino, t_seg



if __name__ == "__main__":

    #dataset = ChallengeData(angular_range = 90, include_fbp=True, random_start_angle=True)
    
    #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

    #ax1.imshow(dataset[0][1].numpy()[0,:,:], cmap="gray")

    dataset = FullChallengeData(include_fbp=True)

    sino, seg, fbp = dataset[2]
    print(sino.shape)
    import matplotlib.pyplot as plt 
    
    start_angle = np.random.randint(low=0, high=721 - angular_range_to_idx[60])
    stop_angle = start_angle + angular_range_to_idx[60]

    sino[0, :start_angle, :] = 0.25*sino[0, :start_angle, :]
    sino[0, stop_angle:, :] = 0.25*sino[0, stop_angle:, :]

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1.imshow(sino[0,:,:].T, cmap="gray")
    ax1.vlines(start_angle, 0, 560, color="r")
    ax1.vlines(stop_angle, 0, 560,color='r')
    ax1.axis("off")

    ax2.imshow(fbp[0,:,:], cmap="gray")
    ax2.axis("off")

    ax3.imshow(seg[0,:,:], cmap="gray")
    ax3.axis("off")

    plt.show()
    