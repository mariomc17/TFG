import torch
import matplotlib.pyplot as plt
import os
import argparse
from diffusers import UNet2DConditionModel, DDIMScheduler
from tqdm import tqdm
from train_diffusion import IMG_SIZE, PhysicsProjector

def parse_args():
    parser = argparse.ArgumentParser(description="Generar galaxia con U-Net de Diffusers.")
    parser.add_argument("--modelo", type=str, default="checkpoints/modelo_epoca_100.pt", help="Ruta al checkpoint")
    parser.add_argument("--escala", type=float, required=True, help="ESCALA_KPC_PX normalizada (0.0 - 1.0)")
    parser.add_argument("--masa", type=float, required=True, help="LOG_MS normalizada (0.0 - 1.0)")
    parser.add_argument("--sfr", type=float, required=True, help="SFR normalizada (0.0 - 1.0)")
    parser.add_argument("--ea", type=float, required=True, help="EA normalizada (0.0 - 1.0)")
    parser.add_argument("--pasos", type=int, default=50, help="Pasos DDIM")
    return parser.parse_args()

def generar_galaxia(modelo_path, escala_norm, masa_norm, sfr_norm, ea_norm, pasos_inferencia):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': torch.set_num_threads(os.cpu_count() or 4)
        
    print(f"Cargando modelo desde {modelo_path}...")
    checkpoint = torch.load(modelo_path, map_location=device, weights_only=False)
    
    vars_entrenadas = checkpoint.get('variables', ['ESCALA_KPC_PX', 'LOG_MS', 'SFR', 'EA'])
    input_dim = len(vars_entrenadas)
    
    print(f"\nGenerando a {IMG_SIZE}x{IMG_SIZE}px en: {device}")
    print(f"Parámetros -> Escala: {escala_norm} | Masa: {masa_norm} | SFR: {sfr_norm} | EA: {ea_norm}")

    unet = UNet2DConditionModel(
        sample_size=IMG_SIZE,
        in_channels=3,
        out_channels=3,
        cross_attention_dim=256,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 1024), 
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    
    projector = PhysicsProjector(input_dim=input_dim).to(device)

    unet.load_state_dict(checkpoint['unet'])
    projector.load_state_dict(checkpoint['projector'])
    
    unet.eval()
    projector.eval()
    
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(pasos_inferencia)

    image = torch.randn((1, 3, IMG_SIZE, IMG_SIZE)).to(device)
    
    valores_usuario = [escala_norm, masa_norm, sfr_norm, ea_norm]
    phys_vector = torch.tensor([valores_usuario], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        encoder_hidden_states = projector(phys_vector)
        for t in tqdm(scheduler.timesteps, desc="Esculpiendo galaxia"):
            noise_pred = unet(image, t, encoder_hidden_states=encoder_hidden_states).sample
            image = scheduler.step(noise_pred, t, image).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1) 
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    
    title_str = " | ".join([f"{v}={val:.2f}" for v, val in zip(vars_entrenadas, valores_usuario)])
    plt.title(f"Galaxia generada\n{title_str}")
    
    plt.savefig("galaxia_generada.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    generar_galaxia(args.modelo, args.escala, args.masa, args.sfr, args.ea, args.pasos)