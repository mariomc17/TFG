import torch
import matplotlib.pyplot as plt
import os
import argparse
from diffusers import DDIMScheduler
from tqdm import tqdm
from mi_u_net import CustomGalaxyUNet
from train_diffusion_mi_u_net import IMG_SIZE, PhysicsProjector

def parse_args():
    parser = argparse.ArgumentParser(description="Generar galaxias condicionadas por física.")
    parser.add_argument("--modelo", type=str, default="checkpoints/modelo_epoca_100.pt", help="Ruta al checkpoint")
    parser.add_argument("--escala", type=float, required=True, help="ESCALA_KPC_PX normalizada (0.0 - 1.0)")
    parser.add_argument("--masa", type=float, required=True, help="Masa normalizada (LOG_MS) (0.0 - 1.0)")
    parser.add_argument("--sfr", type=float, required=True, help="SFR normalizada (0.0 - 1.0)")
    parser.add_argument("--ea", type=float, required=True, help="EA normalizada (0.0 - 1.0)")
    parser.add_argument("--pasos", type=int, default=50, help="Pasos de inferencia DDIM")
    return parser.parse_args()

def generar_galaxia(modelo_path, escala_norm, masa_norm, sfr_norm, ea_norm, pasos_inferencia):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': torch.set_num_threads(os.cpu_count() or 4)
        
    print(f"Generando a {IMG_SIZE}x{IMG_SIZE}px en: {device}")
    print(f"Parámetros -> Escala: {escala_norm} | Masa: {masa_norm} | SFR: {sfr_norm} | EA: {ea_norm}")

    unet = CustomGalaxyUNet(
            n_channels=3,
            n_classes=3,
            embed_dim=256,
            time_dim=256
        ).to(device)
    
    projector = PhysicsProjector(input_dim=4).to(device)

    checkpoint = torch.load(modelo_path, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint['unet'])
    projector.load_state_dict(checkpoint['projector'])
    
    unet.eval()
    projector.eval()
    
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(pasos_inferencia)

    image = torch.randn((1, 3, IMG_SIZE, IMG_SIZE)).to(device)
    phys_vector = torch.tensor([[escala_norm, masa_norm, sfr_norm, ea_norm]], dtype=torch.float32).to(device)

    with torch.no_grad():
        encoder_hidden_states = projector(phys_vector)

        for t in tqdm(scheduler.timesteps, desc="Esculpiendo galaxia"):
            t_batch = torch.tensor([t], dtype=torch.long, device=device)
            noise_pred = unet(x=image, context=encoder_hidden_states, timesteps=t_batch)
            image = scheduler.step(noise_pred, t, image).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1) 
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')

    plt.title(f"Esc={escala_norm:.2f} | M={masa_norm:.2f} | SFR={sfr_norm:.2f} | EA={ea_norm:.2f}")
    nombre_archivo = f"galaxia_esc{escala_norm:.2f}_m{masa_norm:.2f}_sfr{sfr_norm:.2f}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"¡Guardado como {nombre_archivo}!")
    
if __name__ == "__main__":
    args = parse_args()
    generar_galaxia(args.modelo, args.escala, args.masa, args.sfr, args.ea, args.pasos)