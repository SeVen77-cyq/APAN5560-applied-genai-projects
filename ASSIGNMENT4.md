# Assignment 4 setup

Apply this change on top of Assignment 3 (base commit `f3c72d4`). Run the commands from the repository root.

## 1. Set up the local Python environment

```bash
uv venv --python 3.12 .venv-a4
source .venv-a4/bin/activate
uv pip install --python .venv-a4/bin/python 'torch==2.7.1' 'torchvision==0.22.1' tqdm
```

The training scripts pick CUDA first, then Apple Silicon MPS, then CPU. If you hit an MPS error, try `PYTORCH_ENABLE_MPS_FALLBACK=1` or pass `--device cpu`.

## 2. Train both Assignment 4 image models

For a quick integration check on CIFAR-10, run the following once. The first command downloads the dataset, and the second reuses it.

```bash
python -m scripts.train_assignment4 diffusion --epochs 1 --limit 512
python -m scripts.train_assignment4 energy --epochs 1 --limit 512 --sampling-steps 5
```

For the final run, use all 50,000 training images and overwrite the two trial checkpoints. Adjust the batch size if needed.

```bash
python -m scripts.train_assignment4 diffusion --epochs 10 --batch-size 64
python -m scripts.train_assignment4 energy --epochs 10 --batch-size 64
ls -lh models/assignment4_*.pth
```

Both models use RGB CIFAR-10 pixels scaled to [-1,1]. Diffusion learns to predict the Gaussian noise added at a random timestep with MSE. The U-Net uses sinusoidal conditioning and encoder-decoder skip connections. Inference uses DDIM updates on the trained cosine schedule. The EBM learns to score data below negative samples. Its sampler differentiates the summed energy with respect to the images, then takes a clipped noisy gradient-descent step. The outer training loop updates the model weights using the contrastive loss with a squared-energy regularizer.

## 3. Build and test FastAPI in Docker

Open Docker Desktop first. Make sure both trained image checkpoints are present before building.

```bash
docker build -t apan5560-a4 .
docker run --rm -p 8000:8000 apan5560-a4
```

Use a second terminal to run the two requests. The PNG files should open as 128 by 32 pixel RGB grids.

```bash
curl --fail-with-body -sS http://localhost:8000/generate_diffusion \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":4,"diffusion_steps":50}' -o /tmp/a4-diffusion.png

curl --fail-with-body -sS http://localhost:8000/generate_energy \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":4}' -o /tmp/a4-energy.png
```

On macOS, run `open /tmp/a4-diffusion.png /tmp/a4-energy.png`. Check the complete route list at `http://localhost:8000/docs`.

## 4. Commit and push

Once the two final image checkpoints are in place and both endpoints pass in Docker, run:

```bash
git add app/generation.py helper_lib/model.py helper_lib/trainer.py \
  helper_lib/generator.py helper_lib/diffusion.py helper_lib/energy.py \
  scripts/train_assignment4.py Dockerfile .dockerignore \
  .gitignore ASSIGNMENT4.md models/assignment4_diffusion.pth models/assignment4_energy.pth
git diff --cached --check
git commit -m "Add CIFAR-10 diffusion and energy generators"
git push -u origin HEAD
git rev-parse --short HEAD
```

Include the GitHub link and final commit hash with the written theory responses. If you pushed a feature branch, use its link or merge it into the branch the instructor checks.
