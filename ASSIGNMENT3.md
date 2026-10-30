# Assignment 3 / Modules 6 and 7

## Implemented features

- Module 6: reusable GAN model, trainer, and sample generator.
- Assignment 3: specified GAN architecture trained on MNIST for 20 epochs.
- Module 7: word-level LSTM text generation through FastAPI.
- Existing bigram, embedding, and image classification endpoints remain available.

## Main files

- `helper_lib/gan.py`: Generator, Discriminator, GAN, and GAN training loop.
- `helper_lib/model.py`: model factory, including `get_model("GAN")`.
- `helper_lib/trainer.py`: exposes `train_gan`.
- `helper_lib/generator.py`: sample generation for GAN and VAE.
- `helper_lib/lstm.py`: LSTM model and autoregressive text generation.
- `scripts/train_generation.py`: training entry point.
- `app/generation.py`: GAN and LSTM API routes.
- `training/corpus.txt`: LSTM training corpus.

## GAN architecture

Generator:
Noise (B,100) -> Linear(100,6272) -> reshape (B,128,7,7)
-> ConvTranspose2d(128,64,4,2,1) -> BatchNorm2d -> ReLU
-> ConvTranspose2d(64,1,4,2,1) -> Tanh.

Discriminator:
Image (B,1,28,28) -> Conv2d(1,64,4,2,1) -> LeakyReLU(0.2)
-> Conv2d(64,128,4,2,1) -> BatchNorm2d -> LeakyReLU(0.2)
-> Flatten -> Linear(6272,1) -> Sigmoid.

Training computes BCEWithLogitsLoss on the pre-Sigmoid output.
Generator and discriminator use separate Adam optimizers with learning
rate 0.0002 and betas (0.5,0.999). MNIST is normalized to [-1,1].
The batch size is 128; the noise dimension is 100.

## Included inference checkpoints

- `models/assignment3_gan.pth`: trained generator and training metadata.
- `models/module7_lstm.pth`: trained LSTM, vocabulary, and model dimensions.
- `models/assignment2_cnn.pth`: existing classifier checkpoint.

Inference loads the checkpoints; retraining is not required.

## Docker deployment

Run these commands from the repository root:

```bash
docker build -t apan5560-assignment3 .
docker run --rm -p 127.0.0.1:8001:8000 apan5560-assignment3
```

Interactive API documentation: http://localhost:8001/docs

## Query the GAN

```bash
curl --fail-with-body -X POST http://localhost:8001/generate_digits \
  -H "Content-Type: application/json" \
  -d '{"num_samples":25}' \
  --output generated_digits.png
```

Returns a PNG grid. Each generated digit is 28x28 pixels.
The model is unconditional: callers choose the sample count, not the digit.

## Query the LSTM

```bash
curl --fail-with-body -X POST http://localhost:8001/generate_with_rnn \
  -H "Content-Type: application/json" \
  -d '{"start_word":"the","length":20,"temperature":0.8}'
```

Returns JSON with `generated_text` and `model`.
Length includes the starting word. The starting word must be in the
training vocabulary. Unknown words return HTTP 400.

## Reproduce training locally

With uv installed, run from the repository root:

```bash
uv sync
uv run python -m scripts.train_generation gan --epochs 20 --device cpu
uv run python -m scripts.train_generation lstm \
  --corpus training/corpus.txt --epochs 40 --device cpu
```

MNIST downloads automatically. Training overwrites the corresponding
inference checkpoint. The GAN checkpoint contains the generator only;
the training command starts a new run rather than resuming training.

## Validation and limitations

Both local and Docker endpoints were tested successfully:
- GAN: HTTP 200 and a valid 140x140 PNG for 25 samples.
- LSTM: HTTP 200 and a 20-word generated sequence.

Example outputs are included under `outputs/`.
Some generated digits remain distorted. The LSTM uses the small corpus
from the earlier bigram activity, so vocabulary and fluency are limited.
