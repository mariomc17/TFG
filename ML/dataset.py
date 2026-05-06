import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import numpy as np
import json
from PIL import Image

class GalaxiasFisicasDataset(Dataset):
    def __init__(self, hdf5_path, img_size, variables_elegidas):
        self.hdf5_path = hdf5_path
        self.img_size = img_size
        self.variables_elegidas = variables_elegidas
        self.h5_file = None
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = len(f['images'])
            self.stats = json.loads(f.attrs['stats'])
            self.columnas_fisicas = json.loads(f.attrs['columnas_fisicas'])
            
        print(f"Dataset HDF5 enganchado: {self.length} galaxias listas para procesar.")
            
        self.indices_vars = []
        self.indices_errs = []
        
        for var in self.variables_elegidas:
            idx_var = self.columnas_fisicas.index(var)
            self.indices_vars.append(idx_var)
            
            err_col = f"{var}_ERR"
            if err_col in self.columnas_fisicas:
                self.indices_errs.append(self.columnas_fisicas.index(err_col))
            else:
                self.indices_errs.append(-1)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def _normalize(self, value, name):
        v_min = self.stats[name]['min']
        v_max = self.stats[name]['max']
        return (value - v_min) / (v_max - v_min + 1e-8)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            
        img_np = self.h5_file['images'][idx]
        image = Image.fromarray(img_np)
        image = self.transform(image)
        
        phys_data = self.h5_file['fisica'][idx]
        
        vect_final = []
        for i, var_name in enumerate(self.variables_elegidas):
            val = phys_data[self.indices_vars[i]]
            idx_err = self.indices_errs[i]
            
            if idx_err != -1:
                err = phys_data[idx_err]
                val = np.random.normal(loc=val, scale=err)
                
            val_norm = self._normalize(val, var_name)
            vect_final.append(val_norm)
            
        fisica_vector = torch.tensor(vect_final, dtype=torch.float32)
        fisica_vector = torch.clamp(fisica_vector, 0.0, 1.0)
        
        # rgb_np = self.h5_file['rgb'][idx]
        # rgb_vector = torch.tensor(rgb_np, dtype=torch.float32)
        
        return image, fisica_vector