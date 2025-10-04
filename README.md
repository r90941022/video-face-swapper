# Video Changer

智能影片換臉系統 - 自動辨識主要人物並進行高品質臉部替換

Intelligent video face swapping system with automatic dominant face detection and GPU acceleration.

## 📋 目錄 / Table of Contents

- [快速開始 Quick Start](#快速開始-quick-start)
- [功能特色 Features](#功能特色-features)
- [系統需求 Requirements](#系統需求-system-requirements)
- [安裝指南 Installation](#安裝指南-installation)
- [使用說明 Usage](#使用說明-usage)
- [配置說明 Configuration](#配置說明-configuration)
- [效能表現 Performance](#效能表現-performance)
- [技術架構 Architecture](#技術架構-technical-architecture)
- [常見問題 FAQ](#常見問題-faq)
- [故障排除 Troubleshooting](#故障排除-troubleshooting)

---

## 🚀 快速開始 Quick Start

### 🎯 最簡單的方式 - 一鍵執行 / Easiest Way - One Command

```bash
# 1. 將影片放入 input/ 目錄
cp your_video.mp4 input/

# 2. 執行 run.sh (自動處理所有影片和人臉)
./run.sh
```

**完成！** 輸出影片在 `output/final/` 目錄

**特點:**
- ✅ 自動掃描 input/ 裡的所有影片
- ✅ 自動套用 target_images/ 裡的所有人臉
- ✅ 自動產生所有組合的換臉影片
- ✅ 不需要輸入任何參數或檔案名稱

---

### 📋 進階方式 - 手動指定 / Advanced - Manual Specification

如果需要精確控制處理哪個影片和人臉:

```bash
# 激活環境
source venv/bin/activate

# 執行特定組合
python main.py --source input/video.mp4 \
               --target target_images/person1 \
               --output output/final/result.mp4 \
               --auto
```

---

## ✨ 功能特色 Features

✅ **智能取樣 Smart Sampling**
- 不只掃描前 100 幀，而是均勻取樣整部影片
- 智能調整取樣策略：短影片密集取樣，長影片間隔取樣

✅ **自動偵測主角 Dominant Face Detection**
- 自動辨識影片中的主要人物
- 基於出現頻率、臉部大小、清晰度的綜合評分

✅ **高品質換臉 High-Quality Swapping**
- 使用 InsightFace inswapper_128 模型
- 自然無違和的換臉效果

✅ **Face Tracking 技術**
- 即使部分 frame 偵測失敗也能維持連續性
- 自動追蹤前一 frame 的臉部位置

✅ **GPU 加速 GPU Acceleration**
- 自動偵測並使用 NVIDIA GPU
- 處理速度提升 10-20 倍

✅ **完整保留原影片品質**
- 維持原 FPS、解析度
- 完整保留音軌

---

## 💻 系統需求 System Requirements

### 硬體需求 Hardware

| 項目 | 最低 | 建議 |
|------|------|------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 8GB | 16GB+ |
| **GPU** | 無 (可用 CPU) | NVIDIA GPU 8GB+ VRAM |
| **儲存** | 10GB 可用空間 | 20GB+ SSD |

### 軟體需求 Software

- **作業系統**: Linux (Ubuntu 20.04+) / Windows 10+ / macOS
- **Python**: 3.8 - 3.11
- **CUDA**: 12.x (GPU 使用者)
- **Driver**: NVIDIA Driver 535+ (GPU 使用者)
- **FFmpeg**: 4.0+

---

## 📦 安裝指南 Installation

### 1. 系統準備

#### Ubuntu/Linux
```bash
# 更新系統
sudo apt-get update
sudo apt-get upgrade

# 安裝 FFmpeg
sudo apt-get install ffmpeg

# 安裝 Python 3.8+
sudo apt-get install python3 python3-pip python3-venv
```

#### 檢查 GPU (可選但建議)
```bash
# 檢查 NVIDIA GPU
nvidia-smi

# 檢查 CUDA 版本
nvcc --version
```

### 2. 建立專案環境

```bash
# 進入專案目錄
cd /home/user/hjchen/video-changer

# 建立虛擬環境
python3 -m venv venv

# 激活虛擬環境
source venv/bin/activate
```

### 3. 安裝 Python 依賴套件

```bash
# 安裝基本套件
pip install -r requirements.txt
```

### 4. 安裝 GPU 支援 (強烈建議)

**對於 CUDA 12.x 系統：**
```bash
pip uninstall -y onnxruntime-gpu onnxruntime
pip install onnxruntime-gpu==1.19.2 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

**驗證 GPU 安裝：**
```bash
python -c "import onnxruntime as ort; print('GPU available:', 'CUDAExecutionProvider' in ort.get_available_providers())"
```

### 5. 準備目錄結構

```bash
# 確認目錄結構
ls -la
# 應該看到：
# input/           - 放置輸入影片
# target_images/   - 放置目標臉部圖片
# output/          - 輸出結果
# src/             - 原始碼
# main.py          - 主程式
# config.yaml      - 配置檔
```

### 6. 測試安裝

```bash
# 執行測試
python main.py --help

# 應該看到程式參數說明
```

---

## 📂 專案結構 Project Structure

```
video-changer/
├── main.py                      # 主程式入口
├── config.yaml                  # 配置檔案
├── requirements.txt             # Python 依賴套件
├── README.md                    # 本文件
│
├── src/                         # 原始碼目錄
│   ├── frame_sampler.py        # 智能 frame 取樣模組
│   ├── face_detector.py        # 臉部偵測模組 (InsightFace)
│   ├── dominant_analyzer.py    # 主角辨識模組
│   ├── face_swapper.py         # 臉部交換核心模組
│   ├── video_processor.py      # 影片處理管道
│   └── gpu_utils.py            # GPU 工具模組
│
├── input/                       # 輸入影片目錄
│   ├── video1.mp4
│   └── video2.mp4
│
├── target_images/              # 目標臉部圖片目錄
│   ├── person1/                # 第一組臉部圖片
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── photo3.jpg
│   └── person2/                # 第二組臉部圖片
│       ├── photo1.jpg
│       └── photo2.jpg
│
├── output/                     # 輸出目錄
│   ├── final/                  # 最終輸出影片
│   │   └── result.mp4
│   └── analysis/               # 分析結果 (可選)
│       ├── preview_comparison.jpg
│       └── dominant_face_crops/
│
└── venv/                       # Python 虛擬環境
```

## Usage

### Basic Usage

1. **Prepare your inputs:**
   - Place source video in `input/` directory
   - Place 3-5 target face photos in `target_images/` directory
     - Use different angles (front, left, right) for best results
     - Photos should be clear and well-lit
     - Face should be clearly visible

2. **Run in step-by-step mode (recommended for first time):**
```bash
python main.py \
  --source input/source_video.mp4 \
  --target target_images/ \
  --output output/final/result.mp4
```

3. **Or run in automatic mode:**
```bash
python main.py \
  --source input/source_video.mp4 \
  --target target_images/ \
  --output output/final/result.mp4 \
  --auto
```

### Step-by-Step Mode

The default mode shows you the results at each stage and waits for confirmation:

1. **Frame Sampling**: View sampled frames grid
2. **Face Detection**: View detected faces with bounding boxes
3. **Dominant Face Analysis**: View identified dominant face crops and statistics
4. **Target Face Preparation**: Confirm target faces extracted correctly
5. **Preview**: View side-by-side comparison of 10 sample frames
6. **Full Processing**: Process entire video

At each step, you can:
- Press `y` to continue
- Press `n` to stop and adjust settings
- Press `q` to quit

### Configuration

Edit `config.yaml` to customize:

**Frame Sampling:**
```yaml
frame_sampling:
  target_samples: 200              # Number of frames to analyze
  min_interval_seconds: 1.5        # Sampling interval for long videos
```

**Dominant Face Detection:**
```yaml
dominant_face:
  weight_frequency: 0.4            # Weight for appearance frequency
  weight_size: 0.4                 # Weight for face size
  weight_clarity: 0.2              # Weight for face quality
```

**Face Processing:**
```yaml
face_processing:
  face_enhancer: gfpgan_1.4        # Use GFPGAN enhancement
  blend_ratio: 0.75                # Blending strength
```

**Video Output:**
```yaml
video:
  keep_fps: true                   # Preserve original FPS
  keep_audio: true                 # Preserve audio
  output_quality: high             # low, medium, or high
```

**Hardware:**
```yaml
hardware:
  execution_provider: cuda         # cuda, cpu, or coreml (Mac)
```

## How It Works

### 1. Smart Frame Sampling
Instead of analyzing only the first 100 frames, the system:
- Calculates video duration
- Samples frames uniformly across the entire video
- For short videos (<60s): samples densely
- For long videos (≥60s): samples every 1-2 seconds
- Ensures at least 200 frames for robust analysis

### 2. Dominant Face Detection
The system identifies the main person by:
- Detecting all faces in sampled frames
- Tracking unique identities using face embeddings (cosine similarity)
- Scoring each identity based on:
  - **Frequency** (40%): How often they appear
  - **Size** (40%): Average face size in frames
  - **Clarity** (20%): Detection confidence
- Selecting the highest-scoring identity as dominant

### 3. Face Swapping
For each frame in the video:
- Detect all faces
- Match faces to dominant identity using embeddings
- Swap matched faces with target face using InsightFace inswapper
- Optional: Enhance face quality with GFPGAN
- Blend seamlessly with original frame

### 4. Video Reconstruction
- Maintains original resolution, FPS, and quality
- Preserves audio track
- Uses H.264 encoding with configurable quality

## Target Face Guidelines

For best results, provide 3-5 photos of the target face with:
- **Variety of angles**: front, left profile, right profile
- **Good lighting**: well-lit, no harsh shadows
- **Clear face**: face should be clearly visible
- **High resolution**: at least 512x512 pixels
- **Neutral expression**: works best, but not required

## Output Files

After processing, you'll find:

**Analysis Results** (`output/analysis/`):
- `sampled_frames_grid.jpg` - Grid of sampled frames
- `face_tracking_visualization.jpg` - Tracked faces across frames
- `dominant_face_report.txt` - Detailed statistics
- `dominant_face_crops/` - Crops of dominant face from various frames
- `preview_comparison.jpg` - Before/after comparison

**Processed Videos** (`output/final/`):
- Your final video with replaced faces

**Debug Files** (`output/detected_faces/`, `output/sampled_frames/`):
- Individual frames with annotations (if debug enabled)

## Performance

Processing speed depends on:
- Video resolution and length
- Hardware (GPU vs CPU)
- Face enhancement enabled/disabled

**Typical speeds (with GPU):**
- 720p video: ~10-15 fps
- 1080p video: ~8-12 fps
- 4K video: ~3-5 fps

## Troubleshooting

**"No face detected in target images"**
- Ensure target images contain clear, visible faces
- Try images with better lighting
- Face should occupy at least 20% of image

**"Cannot open video"**
- Check video file is not corrupted
- Ensure FFmpeg is installed
- Try converting video to MP4 format

**"Out of memory"**
- Reduce `batch_size` in config
- Process on CPU instead of GPU
- Use lower `output_quality`

**Wrong person selected as dominant**
- Check `output/analysis/dominant_face_report.txt`
- Adjust weights in config (increase `weight_frequency`)
- Ensure video has one clearly dominant person

## Technical Details

**Models Used:**
- **Face Detection**: InsightFace Buffalo_L
- **Face Recognition**: ArcFace (512-dim embeddings)
- **Face Swapping**: InsightFace inswapper_128.onnx
- **Face Enhancement**: GFPGAN v1.4 (optional)

**Similarity Threshold:**
- Face matching uses cosine similarity of ArcFace embeddings
- Default threshold: 0.4 (adjustable in code)

## License

This project uses several open-source components:
- InsightFace: MIT License
- GFPGAN: Apache 2.0 License

## Disclaimer

This tool is for research and educational purposes. Users are responsible for ensuring they have appropriate rights and permissions for any content they process.

## Credits

Built using:
- [InsightFace](https://github.com/deepinsight/insightface)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- OpenCV, NumPy, and other open-source libraries
