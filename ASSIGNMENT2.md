# Assignment 2 — CNN Implementation and API
Author: Yuqi Cheng

## Files
- `helper_lib/assignment2_model.py`: required 64x64 CNN architecture
- `train_assignment2.py`: CIFAR10 training and evaluation
- `models/assignment2_cnn.pth`: trained model weights
- `app/image_classifier.py`: POST /classify endpoint
- `app/main.py`: existing API with classification router
- `results/assignment2_metrics.json`: test metrics
- `results/assignment2_training.log`: training log
- `results/assignment2_api_test.json`: API test result
- `results/cifar10_test_0.png`: sample test image

Theory answers are submitted separately in Assignment2_Theory.pdf.

## Architecture
RGB input (3x64x64)
→ Conv2d(3,16,3,stride=1,padding=1) → ReLU → MaxPool2d(2,2)
→ Conv2d(16,32,3,stride=1,padding=1) → ReLU → MaxPool2d(2,2)
→ Flatten (8192 features)
→ Linear(8192,100) → ReLU → Linear(100,10).

The final layer returns logits.

## Training
All 50,000 CIFAR10 training images are used.
Evaluation uses the 10,000 test images.
Images are resized to 64x64 and converted to tensors.
The API uses the same preprocessing.

Adam optimizer; learning rate 0.001; batch size 64; 5 epochs; seed 42.
Observed test accuracy: 64.61%.
Observed test loss: 1.0090.
Results may vary between runs and devices.

## Setup
Run from the repository root with uv installed:

```bash
uv sync
uv run python -m spacy download en_core_web_md
```

The existing embedding endpoint requires en_core_web_md.

## Train the classifier
```bash
uv run python train_assignment2.py
```

The script downloads CIFAR10 if needed and saves weights and metrics.
Included model weights allow inference without retraining.

## Start the API
```bash
uv run python -m uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000/docs.
Expand POST /classify, click Try it out, upload an image, and click Execute.

Alternatively, run in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F "file=@results/cifar10_test_0.png"
```

The response contains predicted_class, class_index, confidence,
and probabilities for all 10 CIFAR10 classes.
Confidence is the highest softmax probability.

## API verification
The saved test returned HTTP 200.
True class: cat. Predicted class: cat.
Cat probability: approximately 0.7606.
See results/assignment2_api_test.json for the recorded response.

## Docker deployment
With Docker running, execute from the repository root:

```bash
docker build -t assignment2-api .
docker run --rm -p 8001:8000 assignment2-api
```

Open http://127.0.0.1:8001/docs to query POST /classify.

Alternatively, from another terminal in the repository root:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8001/classify \
  -F "file=@results/cifar10_test_0.png"
```

The image includes the API dependencies, spaCy model, helper library,
and trained classifier weights. No host Python environment or retraining
is needed for Docker inference. Internet access is required during build
to download dependencies.

Docker inference was tested successfully and returned HTTP 200 with
predicted_class "cat". Evidence: results/assignment2_docker_test.json.
