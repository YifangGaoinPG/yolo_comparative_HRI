
## Quick Start 🚀 

### 1. Install Dependencies

```
for YOLOv13:
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov13 python=3.11
conda activate yolov13
pip install -r requirements.txt
pip install -e .
```
YOLOv13 suppports Flash Attention acceleration.

for YOLOv9-12:
```
pip install -q git+https://github.com/sunsmarterjie/yolov12.git roboflow supervision flash-attn
```

### 2. Dataset
See dataset folder or use the following code:
import pillow_heif

def mock_register_avif_opener(*args, **kwargs):
    pass

pillow_heif.register_avif_opener = mock_register_avif_opener

from roboflow import Roboflow

ROBOFLOW_API_KEY = 'fill_your_key_here'

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("computer-vision-cuvyr").project("tsl-detection")
version = project.version(2)
dataset = version.download("yolov8")


### 3. Training

Use the following code to train the YOLO models. Make sure to replace `yolov13n.yaml` with the desired model configuration file path, and `coco.yaml` with your coco dataset configuration file.
```python
from ultralytics import YOLO

model = YOLO('yolov13n.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
  device="0,1,2,3",
  workers=16,
)

```
### 4. Validation
[`YOLOv13-N`](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt)
[`YOLOv13-S`](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt)
[`YOLOv13-L`](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt)
[`YOLOv13-X`](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt)

Use the following code to validate the YOLOv13 models on the TSL-Detection dataset. Make sure to replace `{n/s/l/x}` with the desired model scale (nano, small, plus, or ultra).
```python
from ultralytics import YOLO

model = YOLO('yolov13{n/s/l/x}.pt')  # Replace with the desired model scale
metrics = model.val('path_to_dataset_yaml_file')
```

## Related Projects 🔗

- The code is based on [Ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!


