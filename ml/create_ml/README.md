# Create ML → Core ML (Music Vision)

Train a **MobileNet-class Image Classifier** on macOS with Apple Create ML, then
run it on iOS through `CoreMLModelBridge` (Neural Engine via Vision).

Google **ML Kit** (`.tflite`) remains available as `VisionInferenceRuntime.mlKit`.

## 1. Prepare class folders

```bash
python3 ml/prepare_coreml_model.py --labels ml/labels.example.txt
```

This creates:

```
ml/create_ml/TrainingData/<label>/
ml/create_ml/TestingData/<label>/   # optional hold-out images
```

Put at least ~30–50 photos per label (instrument, vinyl, guitar, …).

## 2. Train in Create ML

1. Open **Create ML** (from Xcode → Open Developer Tool → Create ML).
2. New Document → **Image Classification**.
3. Training Data → select `ml/create_ml/TrainingData`.
4. Optional Testing Data → `ml/create_ml/TestingData`.
5. Train. Prefer a MobileNet-style / compact model for phones and accessibility.
6. Export → save as:

```
ios/MusicApp/Models/music_vision_mobilenet.mlpackage
```

## 3. Wire into MusicApp

1. Add `music_vision_mobilenet.mlpackage` to the Xcode target.
2. Add Swift sources under `MusicApp/CoreML/` and `MusicApp/MLKit/`.
3. Default runtime is Core ML:

```swift
let classifier = MusicVisionClassifier(runtime: .coreML) // Create ML model
// let classifier = MusicVisionClassifier(runtime: .mlKit) // TFLite via ML Kit
classifier.prepare()
```

## 4. Convert from TensorFlow instead

If you already have a Keras / SavedModel MobileNet:

```bash
pip install 'coremltools>=7' tensorflow pillow
python3 ml/convert_to_coreml.py \
  --input path/to/mobilenet_saved_model \
  --labels ml/labels.example.txt \
  --output ios/MusicApp/Models/music_vision_mobilenet.mlpackage
```

Keep label order identical to ML Kit / Raspberry Pi EfficientNet exports.
