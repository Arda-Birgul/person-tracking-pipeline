# Person Tracking Pipeline

An advanced person detection and tracking system using YOLO and optimized tracking algorithms with multi-processing support.

## Features

- **Advanced Person Detection**: YOLO-based person detection with configurable confidence thresholds
- **Stable ID Tracking**: Enhanced CentroidTracker for consistent person ID assignment
- **Performance Optimized**: 
  - FP16 optimization and model warmup
  - Multi-processing support with auto-tuning
  - Batch inference and threaded I/O
  - GPU/CPU adaptive processing
- **Visual Enhancement**:
  - Person bounding boxes with ID labels
  - Head detection and visualization
  - Real-time confidence scoring
- **Export Capabilities**: ONNX model export support
- **Monitoring**: Performance profiling and detailed logging

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended) or CPU
- Sufficient RAM for video processing

### Dependencies

The system will automatically install required packages, or install manually:

```bash
pip install ultralytics numpy opencv-python-headless requests torch psutil
```

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd person-tracking-pipeline
```

2. Create input and output directories:
```bash
mkdir InputData OutputData
```

3. Place your video files in the `InputData` directory

## Usage

### Basic Usage

Process all videos in InputData folder with automatic optimization:

```bash
python person_tracker.py
```

### Advanced Usage Examples

#### Manual Worker Count
```bash
# Use 4 workers for processing
python person_tracker.py --workers 4
```

#### Disable Auto-tuning
```bash
# Skip performance auto-tuning and use simple worker allocation
python person_tracker.py --auto-tune false --workers 2
```

#### Custom Model
```bash
# Use a different YOLO model
python person_tracker.py --model yolov8l.pt
```

#### Performance Profiling
```bash
# Enable detailed performance profiling
python person_tracker.py --profile
```

#### ONNX Export
```bash
# Export model to ONNX format
python person_tracker.py --export-onnx
```

#### Detection Interval Adjustment
```bash
# Run detection every 5 frames (default: 3)
python person_tracker.py --detect-interval 5
```

#### Skip Package Installation
```bash
# Don't auto-install packages
python person_tracker.py --no-install
```

#### Batch Size Configuration
```bash
# Use larger batch size for better GPU utilization
python person_tracker.py --batch-size 8
```

### Complete Configuration Example

```bash
python person_tracker.py \
    --workers 6 \
    --model yolov8x.pt \
    --detect-interval 2 \
    --batch-size 6 \
    --profile \
    --export-onnx
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-install` | flag | False | Skip automatic package installation |
| `--workers` | int | 0 (auto) | Number of worker processes (0 = auto-tune) |
| `--auto-tune` | str | "true" | Enable performance auto-tuning |
| `--model` | str | "yolov8m.pt" | YOLO model to use |
| `--no-gpu-cache-clean` | flag | False | Disable GPU cache cleaning |
| `--export-onnx` | flag | False | Export model to ONNX format |
| `--detect-interval` | int | 3 | Run detection every N frames |
| `--batch-size` | int | 4 | Batch inference size |
| `--profile` | flag | False | Enable performance profiling |

## Configuration

Key parameters can be modified at the top of the script:

```python
# Detection thresholds
YOLO_MIN_CONF = 0.25          # Minimum confidence for YOLO detection
MIN_CONF_TO_CREATE = 0.65     # Minimum confidence to create new track
MIN_CONF_TO_SHOW = 0.65       # Minimum confidence to display track

# Tracking parameters
DIST_THRESH = 40.0            # Maximum distance for track association
MAX_MISSED = 6                # Maximum missed detections before track deletion
MIN_HITS = 2                  # Minimum hits before track confirmation

# Processing parameters
DETECT_EVERY_K_FRAMES = 3     # Detection frequency
BATCH_SIZE = 4                # Inference batch size
INFER_IMG_SZ = 1280          # Inference image size
```

## Directory Structure

```
project-root/
├── person_tracker.py          # Main script
├── InputData/                 # Place video files here
│   ├── video1.mp4
│   ├── video2.webm
│   └── ...
├── OutputData/               # Processed results
│   ├── video1/
│   │   ├── Bbox.webm        # Processed video with bounding boxes
│   │   └── log.txt          # Tracking log
│   └── video2/
│       ├── Bbox.webm
│       └── log.txt
└── README.md
```

## Supported Video Formats

- MP4 (`.mp4`)
- WebM (`.webm`)
- AVI (`.avi`)
- MOV (`.mov`)
- MKV (`.mkv`)
- M4V (`.m4v`)

## Output

For each processed video, the system generates:

1. **Processed Video** (`Bbox.webm`): Video with person bounding boxes, IDs, and confidence scores
2. **Tracking Log** (`log.txt`): Detailed frame-by-frame tracking information

### Log Format
```
Frame/Remaining // ID // Status // Confidence%
0/1500 // 1 // Status: İnsan // %: 87
1/1499 // 1 // Status: İnsan // %: 89
```

## Performance Optimization

### Auto-tuning
The system automatically determines optimal worker count based on:
- Available CPU cores
- GPU availability
- System load testing
- Memory constraints

### Manual Optimization Tips

1. **GPU Users**: Use larger batch sizes (8-16) for better GPU utilization
2. **CPU Users**: Limit workers to prevent memory issues
3. **Large Videos**: Increase detection interval to reduce processing time
4. **Accuracy Priority**: Use smaller detection intervals (1-2 frames)

## Troubleshooting

### Common Issues

**GPU Memory Error**:
```bash
# Reduce batch size or enable cache cleaning
python person_tracker.py --batch-size 2 --no-gpu-cache-clean false
```

**Slow Processing**:
```bash
# Increase detection interval or reduce worker count
python person_tracker.py --detect-interval 5 --workers 2
```

**Package Installation Issues**:
```bash
# Install packages manually
pip install ultralytics numpy opencv-python-headless requests torch psutil
python person_tracker.py --no-install
```

**Model Loading Error**:
```bash
# Use lighter model
python person_tracker.py --model yolov8n.pt
```

### System Requirements by Model

| Model | GPU Memory | Processing Speed | Accuracy |
|-------|------------|------------------|----------|
| yolov8n.pt | 2GB+ | Fast | Good |
| yolov8s.pt | 3GB+ | Fast | Good |
| yolov8m.pt | 4GB+ | Medium | Better |
| yolov8l.pt | 6GB+ | Slow | Best |
| yolov8x.pt | 8GB+ | Slowest | Best |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- OpenCV for computer vision operations
- PyTorch for deep learning framework