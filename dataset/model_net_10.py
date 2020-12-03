import glob
import open3d as o3d
import numpy as np

from torch.utils.data import Dataset

class ModelNet10(Dataset):
    def __init__(self,
                 basedir,
                 num_points,
                 partition='train'):

        self.basedir = basedir
        self.categories = ['bed', 'table', 'bathtub', 'chair','desk','dresser', 'monitor', 'night_stand', 'sofa', 'toilet']
        self.filepaths = []
        self.category_idxs  = []
        self.category_labels  = []
        self.num_points = num_points

        for idx, category in enumerate(self.categories):
            paths = glob.glob(os.path.join(self.basedir, category, partition.lower(), '*.off'))

            self.category_idxs += [idx] * len(paths)
            self.category_labels.append(category)

            self.filepaths += paths

    def __len__(self):
        return len(self.category_idxs)

    def load_point_cloud_from_mesh(self, infile):
        mesh = o3d.io.read_triangle_mesh(infile, print_progress=False)
        pc = mesh.sample_points_uniformly(self.num_points)
        return pc.points

    def normalize_points(self, point_cloud):
        points = np.array(point_cloud).astype('float32')
        points = (points - np.mean(points, axis=0)) / (np.std(points,axis=0) + 1e-6)
        return points

    def __getitem__(self, index):
        category = np.array([self.category_idxs[index]])
        data = self.load_point_cloud_from_mesh(self.filepaths[index])
        data = self.normalize_points(data)
        return data, category

# data_set = ModelNet10(2048, partition='test')
# for data, label in data_set:
#     print(type(data), type(label))
#     print(data.shape, label.shape)
#     print(data, label)
#     break
