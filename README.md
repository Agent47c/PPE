# Object Tracking System

A real-time object detection and tracking system using YOLOv8 and StrongSort. This application detects and tracks persons and PPE (Personal Protective Equipment) in video streams with multi-threaded processing for optimal performance.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for detecting objects including persons and PPE
- **Multi-object Tracking**: Implements StrongSort algorithm for robust tracking across frames
- **Multi-threaded Processing**: Separates video reading, detection, tracking, and rendering into independent threads for better performance
- **FPS Monitoring**: Real-time FPS calculation and display
- **Person-specific Tracking**: Dedicated tracking for class ID 11 (persons) with unique IDs
- **PPE Detection**: Detects and displays all non-person objects with class labels
- **Frame Counting**: Monitors total frames processed

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- PyTorch (`torch`)
- Ultralytics YOLO (`ultralytics`)
- BoxMOT (`boxmot`)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install opencv-python torch ultralytics boxmot numpy
```

3. Download the required models:
   - Place your YOLOv8 model weights in the specified model path (e.g., `D://Testing_Models//FocalLoss//best.pt`)
   - Download the StrongSort ReID weights from [BoxMOT](https://github.com/mikel-brostrom/yolo_tracking) and place them at `D:/PPE Detection/osnet_x0_25_msmt17.pt`

## Usage

### Basic Example

```python
from ObjectTracking import Main_App

# Initialize the application
app = Main_App(
    Video_path="your_video.mp4",
    Model_path="path/to/your/model.pt",
    QueueSize=5
)

# Start processing
app.run()
```

### Parameters

- **Video_path** (str): Path to the input video file
- **Model_path** (str): Path to the YOLOv8 model weights
- **QueueSize** (int): Maximum size of processing queues for each pipeline stage

### Running the Application

```bash
python ObjectTracking.py
```

**Keyboard Controls:**
- Press `1` to stop the application early

## Architecture

### Components

#### VideoLoader
Handles video file loading and frame extraction using OpenCV.

#### YOLODetector
Loads and runs YOLOv8 inference for object detection with configurable confidence threshold.

#### ObjectTracker
Uses StrongSort algorithm for multi-object tracking with ReID (Re-Identification) features.

#### FPSCounter
Calculates real-time frames per second for performance monitoring.

#### Main_App
Orchestrates the entire pipeline with multi-threaded processing:
- **VideoFrameReader**: Reads frames from video file
- **ObjectDetection**: Detects objects using YOLO
- **ObjectTracking**: Tracks persons (class ID 11) using StrongSort
- **BoundingBox**: Renders bounding boxes and labels on frames

### Processing Pipeline

```
Video Input
    ↓
VideoFrameReader (Thread 1)
    ↓ (frame_queue)
ObjectDetection (Thread 2)
    ↓ (det_queue)
ObjectTracking (Thread 3)
    ↓ (track_queue)
BoundingBox Rendering (Thread 4)
    ↓
Display Output
```

## Output

The application displays:
- **Tracked Persons**: Blue bounding boxes with Person ID
- **PPE Objects**: Green bounding boxes with class labels
- **FPS**: Current frames per second in top-left corner
- **Frame Count**: Total frames processed

## Configuration

### Confidence Threshold
Adjust detection confidence in `YOLODetector`:
```python
self.Detector = YOLODetector(Model_path, confidence=0.1)
```

### Device Selection
Change device in `ObjectTracker` (currently set to CPU):
```python
device="cpu"  # Change to "cuda" for GPU acceleration
```

### Queue Size
Adjust queue size based on your system capabilities:
```python
app = Main_App(..., QueueSize=5)  # Increase for faster systems
```

## Performance Tips

1. **GPU Acceleration**: Change device to "cuda" in ObjectTracker for GPU inference
2. **Queue Size**: Adjust QueueSize parameter based on available memory
3. **Model Size**: Use smaller YOLOv8 variants (nano, small) for faster inference
4. **Confidence Threshold**: Increase threshold to reduce false positives and improve speed

## Troubleshooting

### Video file not found
Ensure the video path is correct and the file exists:
```
❌ Could not open video file: path/to/video
```

### Model loading issues
Verify model path and ensure the .pt file is a valid PyTorch model:
```
❌ YOLO model failed to load
```

### Memory issues
Reduce QueueSize or use a smaller model variant

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add your contact information here]

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BoxMOT](https://github.com/mikel-brostrom/yolo_tracking)
- [OpenCV](https://opencv.org/)
