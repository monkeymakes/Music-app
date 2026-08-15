# Music-app

Share music with others.

## Cloud ML Vision

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | TinyML / MobileNet | Core ML / ML Kit |
| **Raspberry Pi** | EfficientNet | TFLite |
| **Cloud / server** | **EfficientNet** | **TensorFlow Serving** |

### TensorFlow Serving

```bash
pip install -r ml/requirements-serving.txt
python3 ml/prepare_serving_model.py --export-stub
docker compose -f serving/docker-compose.yml up
python3 serving/client/predict_image.py --image path/to/photo.jpg
```

See [`serving/README.md`](serving/README.md).
