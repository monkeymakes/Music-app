# Place versioned SavedModels here for TensorFlow Serving.

# Layout:
#   music_vision/
#     1/          ← version 1 (SavedModel: saved_model.pb + variables/)
#     2/          ← optional newer version
#   music_vision_labels.txt
#   music_vision_serving_manifest.json

# Generate sidecars + optional stub:
#   python3 ml/prepare_serving_model.py --export-stub
