import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm
import os
from dataset import GalaxiasFisicasDataset 

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# HDF5_PATH = os.path.join(REPO_ROOT, "h5_sin_rgb", "dataset_galaxias_sin_rgb.h5")
HDF5_PATH = "dataset_galaxias_sin_rgb.h5"
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

VARS_ELEGIDAS = ['ESCALA_KPC_PX', 'LOG_MS', 'SFR', 'EA']
###########################################################################################
    
class PhysicsProjector(nn.Module):
    def __init__(self, input_dim, embed_dim=256, seq_len=4):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim * seq_len)
        )
        
    def forward(self, x):
        projected = self.net(x)
        return projected.view(-1, self.seq_len, self.embed_dim)

def main():
    input_dim = len(VARS_ELEGIDAS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando entrenamiento en: {device} | Imagen: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Variables seleccionadas: {VARS_ELEGIDAS}")

    dataset = GalaxiasFisicasDataset(hdf5_path=HDF5_PATH, img_size=IMG_SIZE, variables_elegidas=VARS_ELEGIDAS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    unet = UNet2DConditionModel(
        sample_size=IMG_SIZE,
        in_channels=3,
        out_channels=3,
        cross_attention_dim=256, 
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 1024), 
        
        down_block_types=(
            "DownBlock2D",          
            "DownBlock2D",          
            "DownBlock2D",            
            "CrossAttnDownBlock2D",   
            "CrossAttnDownBlock2D",    
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",   
            "CrossAttnUpBlock2D",   
            "UpBlock2D",            
            "UpBlock2D",            
            "UpBlock2D",            
        ),
        resnet_time_scale_shift="scale_shift" 
    ).to(device)

    projector = PhysicsProjector(input_dim=input_dim).to(device)

    optimizer = AdamW(list(unet.parameters()) + list(projector.parameters()), lr=LR)
    criterion = nn.MSELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        unet.train()
        projector.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            clean_images, phys_vectors = batch 
            clean_images = clean_images.to(device)
            phys_vectors = phys_vectors.to(device)

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            encoder_hidden_states = projector(phys_vectors)
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
        
        print(f"Época {epoch+1} terminada | MSE Loss Medio: {epoch_loss/len(dataloader):.4f}")

        if (epoch + 1) % 2 == 0:
            torch.save({
                'unet': unet.state_dict(),
                'projector': projector.state_dict(),
                'variables': VARS_ELEGIDAS
            }, f"{CHECKPOINT_DIR}/modelo_epoca_{epoch+1}.pt")

if __name__ == "__main__":
    main()