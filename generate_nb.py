import json

notebook = {
    'cells': [],
    'metadata': {},
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.split('\n')]
    })

def add_code(text):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.split('\n')]
    })

add_md('## 1. Install Dependencies')
add_code('!pip install -q torch torchvision diffusers accelerate tqdm pillow kagglehub torchmetrics[image]')

add_md('## 2. Auto-Setup: Clone Repo & Restore Progress\nThis cell handles setting up the repository and restoring any checkpoints or pre-encoded latents so you don\'t lose progress.')
add_code('''import os
import glob
import shutil

print("--- Starting ELT-SR Auto-Setup ---")

# Clone if missing
if not os.path.exists('/kaggle/working/elt-supervised-7.3m'):
    print("Cloning repository...")
    os.system("git clone https://github.com/ParthBrijpuria/elt-supervised-7.3m.git /kaggle/working/elt-supervised-7.3m")

os.chdir('/kaggle/working/elt-supervised-7.3m')

# Always ensure we have the latest code
os.system("git pull")

# Setup directories
ckpt_target_dir = "checkpoints_64"
os.makedirs(ckpt_target_dir, exist_ok=True)

# Restore checkpoint from Kaggle Input (if attached)
found_ckpts = glob.glob("/kaggle/input/**/elt_sr_*.pt", recursive=True)
if found_ckpts:
    target_path = os.path.join(ckpt_target_dir, os.path.basename(found_ckpts[0]))
    if not os.path.exists(target_path):
        shutil.copy2(found_ckpts[0], target_path)
        print(f"Restored checkpoint from input: {target_path}")

# Restore latents
found_latents = glob.glob("/kaggle/working/elt_cache/**/latents_ffhq_128.pt", recursive=True)
if not found_latents:
    found_latents = glob.glob("/kaggle/input/**/latents_ffhq_128.pt", recursive=True)

if found_latents and not os.path.exists("latents_ffhq_128.pt"):
    shutil.copy2(found_latents[0], "latents_ffhq_128.pt")
    print("Restored latents_ffhq_128.pt")

print(f"\\n✅ Setup Complete! Checkpoints found in {ckpt_target_dir}: {len(glob.glob(f'{ckpt_target_dir}/*.pt'))}")''')

add_md('## 3. Download FFHQ Dataset')
add_code('''import kagglehub
path = kagglehub.dataset_download("greatgamedota/ffhq-face-data-set")
print("Dataset Path:", path)''')

add_md('## 4. Run Training (16x16 -> 64x64)\nThis will automatically resume from `checkpoints_64/elt_sr_latest.pt` if it exists.')
add_code('''!accelerate launch --num_processes 2 run_train.py \\
    --train_dir $path \\
    --img_size 64 \\
    --scale 4 \\
    --output_dir checkpoints_64 \\
    --epochs 1000 \\
    --batch_size 64''')

add_md('## 5. View Visual Samples\nCheck the images generated every 10 epochs during training.')
add_code('''from IPython.display import display, Image
import glob

# Sort by creation time to see the latest first
sample_images = sorted(glob.glob("checkpoints_64/samples/*.png"))

for img in sample_images[-5:]: # Show last 5
    print(img)
    display(Image(filename=img, width=400))''')

add_md('## 6. Run Evaluation (PSNR, SSIM, LPIPS)')
add_code('''!python run_eval.py \\
    --val_dir $path/thumbnails128x128 \\
    --checkpoint checkpoints_64/elt_sr_best.pt \\
    --img_size 64 \\
    --scale 4 \\
    --batch_size 16 \\
    --ddim_steps 50 \\
    --max_batches 100''')

add_md('## 7. Export Best Model\nRun this when you are ready to save the final weights to your Kaggle Output tab for downloading.')
add_code('''import shutil
shutil.copy2("/kaggle/working/elt-supervised-7.3m/checkpoints_64/elt_sr_best.pt", "/kaggle/working/elt_sr_best_64.pt")
print("Saved to /kaggle/working/elt_sr_best_64.pt. Check the Output tab to download!")''')

with open('C:/Users/Parth/Downloads/elt-supervised-7-3m-16to64-CLEAN.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)
