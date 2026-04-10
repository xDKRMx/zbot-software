# Models

This folder is **optional**. The app works without any models (pure OpenCV heuristics).

## Files and expected behavior

- `models/fire_classifier.tflite`
  - **Binary classifier** (fire yes/no).
  - Expected input: RGB image, `224x224` (default; see `AppConfig`), shape `[1, 224, 224, 3]`.
  - Output: either a single score (shape `[1,1]`) or two scores (shape `[1,2]`).
  - The app uses `positive_index=1` by default for `[1,2]` outputs (configure in `AppConfig`).
  - If missing, the system falls back to HSV fire/smoke heuristics.

- `models/object_detector.tflite`
  - **Generic object detector** (optional).
  - Expected to follow common TFLite SSD outputs: boxes, classes, scores, count.
  - If you have labels, add `models/object_labels.txt` with one label per line.
  - If missing, the app ignores this hook.

- `models/deeplabv3_resnet50.pth`
  - Optional semantic segmentation weights (PyTorch).
  - Expected to match torchvision DeepLabV3 ResNet-50 COCO-with-VOC labels.
  - Labels: `models/segment_labels.txt` (one label per line, VOC 21 classes).
  - If missing, the segmentation CLI exits with a clear message.

## Keeping models Pi-friendly

- Prefer TFLite models smaller than ~20MB.
- Quantized (uint8) models are ideal for Raspberry Pi.
- Use `tflite-runtime` on Pi for smaller install footprint.
