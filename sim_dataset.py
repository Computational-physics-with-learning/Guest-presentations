
import os 

import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms.functional import rotate

import numpy as np 
import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw

from scipy.ndimage.morphology import binary_closing


class SimulatedPhantom(Dataset):
    """
    Generates a dataset containing synthesized phantoms.
    WARNING: two calls of getitem produce seperate samples
    """
    def __init__(self,subset, data_generation_methods = ["phantom_1", "phantom_2"], data_generation_probability = [1, 1] , length=100, max_iter=25, rotate=False):
        """
        :param length: legth of dataset (needed to define __len__)
        """

        self.length = length
        self.subset = subset
        self.max_iter = max_iter
        self.rotate = rotate

        assert len(data_generation_probability) == len(data_generation_probability), "Number of data generation methods and probabilities does not match!"

        self.data_generation_methods = data_generation_methods
        self.data_generation_probability = data_generation_probability/np.sum(data_generation_probability)

    def __len__(self):
        return self.length

    def __getitem__(self, IDX):

        disc_level = np.random.uniform(low=0.004, high=0.007)

        method = np.random.choice(self.data_generation_methods, p = self.data_generation_probability)

        if method == "phantom_1":
            phantom = create_phantom(max_iter=self.max_iter)
        if method == "phantom_2":
            phantom = create_phantom2()
        if method == "phantom_3":
            phantom = create_phantom3(max_iter=self.max_iter)
        if method == "phantom_4":
            phantom = create_phantom4()
        if method == "phantom_5":
            phantom = create_phantom5()
            

        phantom = torch.from_numpy(phantom).float()

        if self.rotate:
            angle = np.random.randint(-180, 180)
            phantom = rotate(phantom, angle)


        return phantom, phantom*disc_level



class MmapDataset(Dataset):
    def __init__(self, x_file, y_file, num_samples):
        super(MmapDataset, self).__init__()
        self.x = np.load(x_file, mmap_mode='r')#np.memmap(x_file, mode='r', shape=(num_samples, 3, 96, 96), dtype='float32')
        self.y = np.load(y_file, mmap_mode='r')#np.memmap(y_file, mode='r', shape=(num_samples, 3, 96, 96), dtype='float32')

    def __getitem__(self, item):
        x = torch.from_numpy(np.copy(self.x[item]))
        y = torch.from_numpy(np.copy(self.y[item]))
        return x, y
    def __len__(self):
        return self.x.shape[0]


class SimulatedPhantomCached():
    def __init__(self, batch_size: int = 4, 
                       num_data_loader_workers:int = 8, shuffle=False):

        self.batch_size = batch_size
        self.num_data_loader_workers = num_data_loader_workers 
        self.shuffle = shuffle

        cache_path = "/localdata/htc2022_data/cached_dataset"
        
        self.sim_dataset = MmapDataset(os.path.join(cache_path, 'phantom_seg_2500.npy'), 
                                        os.path.join(cache_path, 'sinogram_gt_2500.npy'), 
                                        2500)

    def get_dataloader(self):

        return DataLoader(self.sim_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=self.shuffle, pin_memory=True) # shuffle=True


"""
Alex:

Create a disk with non-overlapping circular holes.
"""
def create_phantom(min_holes=0, max_holes = 15, max_iter=40):
    xx, yy = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))

    x_center = np.random.rand()*0.045 - 0.0225
    y_center = np.random.rand()*0.045 - 0.0225
    
    phantom = np.zeros((1, 512, 512))

    phantom[:, (xx - x_center)**2 + (yy-y_center)**2 < 0.85] = 1.0

    num_holes = np.random.randint(min_holes, max_holes)

    circle_list = [] 
    iter = 0
    while(len(circle_list) < num_holes):
        # create random center (x,y) and radius r
        x_c = (np.random.rand()*2 - 1)*0.5
        y_c = (np.random.rand()*2 - 1)*0.5
        radius = np.random.rand()*0.15 + 0.05

        collide = False
        for x, y, r in circle_list:
            d = np.sqrt((x_c - x)**2 + (y_c - y)**2)
            if d < np.sqrt(radius + r):
                collide = True
                break

        if not collide:
            circle_list.append((x_c, y_c, radius))

        iter = iter + 1 
        if iter > max_iter:
            break

    for x, y, r in circle_list:            
        phantom[:, (xx - x)**2 + (yy-y)**2 < r**2] = 0

    return phantom

"""
Clemens: 

Create a disk with holes using two sets of lines with different angles.
"""
def create_phantom2(center=None, radius=235, distance=32):
    '''
    Parameters
    ----------
    center : array, optional
        An array with 2 entries specifying the center.
        The default is a random location
    radius : int, optional
        The radius of the disc. The default is 235 (like the challenge data).
    distance : int, optional
        Specifies the distance between the holes. The smaller this value,
        the more holes are created. The default is 32.

    Returns
    -------
    phan : numpy array
        The phantom as a binary image (values zero and one).

    '''
    
    # Start with a black image
    phan = np.zeros([512,512], dtype='uint8')
    if center==None:
        center=255.5+3*np.random.randn(2)
    
    # Choose an angle by random            
    alph = np.pi*np.random.rand()
    
    nbr = int(2*radius/(3*distance))
    d2 = int(distance/2)
    
    for k in range(2):
        # draw white lines (not totally straight) with the selected angle
        for i in np.arange(nbr):
            px = center[0]-radius*np.cos(alph)-(i-(nbr-1)/2)*3*distance*np.sin(alph)
            py = center[1]-radius*np.sin(alph)+(i-(nbr-1)/2)*3*distance*np.cos(alph)
            gamma = alph + 0.02*np.random.randn()
            for j in np.arange(int(20*radius/distance)):
                phan[int(px)-d2:int(px)+d2,int(py)-d2:int(py)+d2]=1
                gamma = alph + 0.98*(gamma-alph) + 0.02*np.random.randn()
                px = px + distance/10*np.cos(gamma)
                py = py + distance/10*np.sin(gamma)
        #choose a second angle and repeat
        alph = alph + np.pi/6 + 2/3*np.pi*np.random.rand()
    

    xx, yy = np.meshgrid(np.linspace(0, 512, 512), np.linspace(0, 512, 512))

    circle_coordinates = (xx - center[0])**2 + (yy-center[1])**2 

    phan[circle_coordinates > radius**2] = 0.0
    phan[((radius-distance)**2 < circle_coordinates)  & (circle_coordinates < radius**2)] = 1.0


    # Do a morphological closing
    mask = np.ones([7,7])
    mask[0,[0,-1]]=0
    mask[-1,[0,-1]]=0
    phan = binary_closing(phan, structure=mask)
    
    return np.expand_dims(phan, 0)

"""
Alex:

Create a disk with generate_polygon.
"""
def create_phantom3(min_holes=3, max_holes = 15, max_iter=60, distance_between=10):
        
    I = np.zeros((512, 512))

    xx, yy = np.meshgrid(np.linspace(0, 512, 512), np.linspace(0, 512, 512))

    x_center = 256 + np.random.randint(-10, 10)
    y_center = 256 + np.random.randint(-10, 10)
        
    I[(xx - x_center)**2 + (yy-y_center)**2 < 235**2] = 1.0


    im = Image.fromarray(np.uint8(I))

    draw = ImageDraw.Draw(im)

    num_forms = np.random.randint(min_holes, max_holes)

    circle_list = [] 
    vertices_list = []
    iter = 0
    while(len(circle_list) < 8):
        avg_radius = np.random.randint(20, 60)
        
        center_x = 256 + np.random.randint(-120, 120)
        center_y = 256 + np.random.randint(-120, 120)

        collide = False
        for x, y, r in circle_list:
            d = (center_x - x)**2 + (center_y - y)**2
            if d < (avg_radius + r + distance_between)**2:
                collide = True
                break

        if not collide:
            num_vertices = np.random.randint(5, 9)
            vertices = generate_polygon(center=(center_x, center_y),
                                avg_radius=avg_radius,
                                irregularity=0.2,
                                spikiness=0.1,
                                num_vertices=num_vertices)
            
            vertices_list.append(vertices)
            circle_list.append((center_x, center_y, avg_radius))
            
        iter = iter + 1 
        if iter > max_iter:
            break

    for vertices in vertices_list:            
        draw.polygon(vertices, fill=0)


    phantom = np.asarray(im)

    mask = np.ones([7,7])
    mask[0,[0,-1]]=0
    mask[-1,[0,-1]]=0
    phantom = binary_closing(phantom, structure=mask)

    return np.expand_dims(phantom, 0)


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

"""
Alex:

Create random forms by taking the level set of sums of gaussians
"""
def create_phantom4():
    xx, yy = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))

        
    phantom = np.zeros((512, 512))


    #pos = np.stack([xx.ravel(), yy.ravel()])
    for j in range(2):
        r = np.random.rand()*0.7
        for i in range(6):
            phi = np.random.rand()*2*np.pi

            x_mean = r*np.cos(phi)#np.random.rand()*1.8 - 0.9
            y_mean = r*np.sin(phi)#np.random.rand()*1.8 - 0.9
            x_std = 40 + np.random.rand()*20
            y_std = 40 + np.random.rand()*20

            cov = 10 + np.random.rand()*10
            phantom +=np.exp(-x_std*(xx - x_mean)**2 - y_std*(yy - y_mean)**2 + cov*(xx - x_mean)*(yy - y_mean))

        
        
    cut_off = np.quantile(phantom.ravel(), 0.7) # 0.75
    phantom[phantom > cut_off] = 1.0
    phantom[phantom <= cut_off] = 0.0

    phantom = 1 - phantom

    center=255.5+3*np.random.randn(2)

    radius=235+3*np.random.randn()
    distance=32

    xx, yy = np.meshgrid(np.linspace(0, 512, 512), np.linspace(0, 512, 512))

    circle_coordinates = (xx - center[0])**2 + (yy-center[1])**2 

    phantom[circle_coordinates > radius**2] = 0.0
    phantom[((radius-distance)**2 < circle_coordinates)  & (circle_coordinates < radius**2)] = 1.0

    return np.expand_dims(phantom, 0) 

"""
Alex:

Create a disk with generate_polygon and generate_polygon for the full object.
"""
def create_phantom5(min_holes=3, max_holes = 15, max_iter=60, distance_between=10):
            
    I = np.zeros((512, 512))
    im = Image.fromarray(np.uint8(I))

    draw = ImageDraw.Draw(im)

    x_center = 256 + np.random.randint(-10, 10)
    y_center = 256 + np.random.randint(-10, 10)

    num_vertices = np.random.randint(3, 13)
    vertices = generate_polygon(center=(x_center, y_center),
                                    avg_radius=220,
                                    irregularity=0.2,
                                    spikiness=0.1,
                                    num_vertices=num_vertices)

    draw.polygon(vertices, fill=1)

    num_forms = np.random.randint(1, 8)

    distance_between=10
    circle_list = [] 
    vertices_list = []
    iter = 0
    while(len(circle_list) < 8):
        avg_radius = np.random.randint(20, 60)

        center_x = 256 + np.random.randint(-120, 120)
        center_y = 256 + np.random.randint(-120, 120)

        collide = False
        for x, y, r in circle_list:
            d = (center_x - x)**2 + (center_y - y)**2
            if d < (avg_radius + r + distance_between)**2:
                collide = True
                break

        if not collide:
            num_vertices = np.random.randint(5, 9)
            vertices = generate_polygon(center=(center_x, center_y),
                                avg_radius=avg_radius,
                                irregularity=0.2,
                                spikiness=0.1,
                                num_vertices=num_vertices)

            vertices_list.append(vertices)
            circle_list.append((center_x, center_y, avg_radius))

        iter = iter + 1 
        if iter > 40:
            break

    for vertices in vertices_list:            
        draw.polygon(vertices, fill=0)

    phantom = np.asarray(im)


    mask = np.ones([7,7])
    mask[0,[0,-1]]=0
    mask[-1,[0,-1]]=0
    phantom = binary_closing(phantom, structure=mask)

    return np.expand_dims(phantom, 0)


def create_phantom_OOD(min_holes=0, max_holes = 15, max_iter=40):
    xx, yy = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))

    x_center = np.random.rand()*0.045 - 0.0225
    y_center = np.random.rand()*0.045 - 0.0225
    
    phantom = np.zeros((1, 512, 512))

    #phantom[:, (xx - x_center)**2 + (yy-y_center)**2 < 0.85] = 1.0
    phantom[:, 64:448, 64:448] = 1.0


    num_holes = np.random.randint(min_holes, max_holes)

    circle_list = [] 
    iter = 0
    while(len(circle_list) < num_holes):
        # create random center (x,y) and radius r
        x_c = (np.random.rand()*2 - 1)*0.5
        y_c = (np.random.rand()*2 - 1)*0.5
        radius = np.random.rand()*0.15 + 0.05

        collide = False
        for x, y, r in circle_list:
            d = np.sqrt((x_c - x)**2 + (y_c - y)**2)
            if d < np.sqrt(radius + r):
                collide = True
                break

        if not collide:
            circle_list.append((x_c, y_c, radius))

        iter = iter + 1 
        if iter > max_iter:
            break

    for x, y, r in circle_list:            
        phantom[:, (xx - x)**2 + (yy-y)**2 < r**2] = 0

    return phantom


if __name__== "__main__":

    import matplotlib.pyplot as plt 
    dataset = SimulatedPhantom(subset="train", rotate=True, data_generation_methods=["phantom_1", "phantom_2", "phantom_3", "phantom_4"],
                data_generation_probability=[2.25, 2.5, 3, 1.1], length=200, max_iter=80)
    fig, axes = plt.subplots(6,6)

    for idx, ax in enumerate(axes.ravel()):
        phantoms_seg, phantoms = dataset[idx]
        ax.imshow(phantoms_seg[0,:,:], cmap="gray")
        ax.axis("off")
    plt.show()

    """
    import matplotlib.pyplot as plt 


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

    phantom = create_phantom(max_iter=50)
    ax1.imshow(phantom[0,:,:], cmap="gray")
    ax1.axis("off")
    ax1.set_title("Random Circles")
    phantom = create_phantom2()
    ax2.imshow(phantom[0,:,:], cmap="gray")
    ax2.axis("off")
    ax2.set_title("Lines From Random Angles")
    phantom = create_phantom3(max_iter=50)
    ax3.imshow(phantom[0,:,:], cmap="gray")
    ax3.axis("off")
    ax3.set_title("Random Polygons")
    phantom = create_phantom4()
    ax4.imshow(phantom[0,:,:], cmap="gray")
    ax4.axis("off")
    ax4.set_title("Gaussian Level Sets")
    plt.show()
    """
    """
    dataset_size = 20000


    dataset = SimulatedPhantom(subset="train", rotate=True, data_generation_methods=["phantom_1", "phantom_2", "phantom_3", "phantom_4"],
                data_generation_probability=[2.25, 2.5, 3, 1], length=dataset_size, max_iter=25)

    
    ### cache dataset
    from numpy.lib.format import open_memmap
    import os 
    from tqdm import tqdm 
    import odl
    from odl.contrib.torch import OperatorModule
    from htc.util.create_ray_transform import get_ray_trafo

    full_ray_trafo = get_ray_trafo(start_angle=0, stop_angle=721)
    full_ray_trafo_torch = OperatorModule(full_ray_trafo)


    base_dir = "/localdata/htc2022_data/cached_dataset"

    flush_interval = 1000
    num_samples = len(dataset)
    
    memmaps_x = open_memmap(os.path.join(base_dir, "phantom_seg_{}.npy".format(dataset_size)), mode='w+',
                                    shape=(num_samples, 1, 512, 512),dtype="float32")
    memmaps_y = open_memmap(os.path.join(base_dir, "sinogram_gt_{}.npy".format(dataset_size)), mode='w+',
                                    shape=(num_samples, 1, 721, 560),dtype="float32")                             
    for k in tqdm(range(num_samples)):            
        
        phantoms_seg, phantoms = dataset[k]

        phantoms_seg = phantoms_seg.unsqueeze(0)
        phantoms = phantoms.unsqueeze(0).to("cuda")

        with torch.no_grad():
            y = full_ray_trafo_torch(phantoms)
            y_noise = y + 0.01*torch.mean(torch.abs(y), dim=[1,2,3], keepdim=True)*torch.randn(y.shape).to(y.device)
            y_noise = torch.clamp(y_noise, 0)

            memmaps_y[k] = y_noise.detach().cpu().numpy()
            memmaps_x[k] = phantoms_seg.detach().cpu().numpy()

            if (k + 1) % flush_interval == 0:
                memmaps_y.flush()
                memmaps_x.flush()

        memmaps_y.flush()
        memmaps_x.flush()
       
    """