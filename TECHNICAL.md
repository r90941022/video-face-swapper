# Technical Documentation
# 技術文件

## Architecture Overview / 架構總覽

### System Components / 系統組件

```
┌─────────────────────────────────────────────────────────┐
│                    Main Pipeline                         │
│                     main.py                              │
└────────────┬────────────────────────────────────────────┘
             │
             ├──> 1. Frame Sampler (frame_sampler.py)
             │    └──> 智能 frame 取樣
             │
             ├──> 2. Face Detector (face_detector.py)
             │    └──> InsightFace buffalo_l 臉部偵測
             │
             ├──> 3. Dominant Analyzer (dominant_analyzer.py)
             │    └──> ArcFace embeddings + 評分系統
             │
             ├──> 4. Face Swapper (face_swapper.py)
             │    └──> inswapper_128 換臉模型
             │
             └──> 5. Video Processor (video_processor.py)
                  └──> FFmpeg 影片重建 + 音訊合併
```

---

## Module Details / 模組詳細說明

### 1. Frame Sampler (frame_sampler.py)

**Purpose / 目的：**
智能取樣 video frames，確保分析涵蓋整部影片而非只有開頭

**Key Features / 主要功能：**
- 根據影片長度自動調整取樣策略
- 短影片（<60秒）：密集取樣
- 長影片（≥60秒）：每 1.5 秒取樣一次
- 目標：取樣約 200 frames 進行分析

**Algorithm / 演算法：**
```python
if duration <= 60:
    # 短影片：密集取樣
    sample_interval = total_frames // target_samples
else:
    # 長影片：固定時間間隔
    sample_interval = int(fps * min_interval_seconds)
```

**Performance / 效能：**
- 8秒影片（192 frames）：約 5-6 秒取樣完成
- 2小時影片（172,800 frames）：約 10-15 秒取樣完成

---

### 2. Face Detector (face_detector.py)

**Purpose / 目的：**
使用 InsightFace 偵測並擷取臉部特徵

**Models Used / 使用模型：**
- **Detection**: InsightFace buffalo_l (det_10g.onnx)
- **Recognition**: ArcFace (w600k_r50.onnx)
- **Landmarks**: 2D/3D landmark models

**Face Embedding / 臉部特徵向量：**
- Dimension: 512-dim vector
- Normalization: L2 normalized
- Similarity Metric: Cosine similarity

**Detection Parameters / 偵測參數：**
```yaml
detection_size: (640, 640)
confidence_threshold: 0.3
min_face_size: 50px
```

**GPU Acceleration / GPU 加速：**
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# GPU 加速約 10-15x faster than CPU
```

**Performance / 效能：**
- GPU: ~50-54 faces/sec
- CPU: ~5-8 faces/sec

---

### 3. Dominant Analyzer (dominant_analyzer.py)

**Purpose / 目的：**
辨識影片中的主要人物

**Scoring Algorithm / 評分演算法：**

```python
total_score = (
    frequency_score * 0.4 +    # 出現頻率 40%
    size_score * 0.4 +         # 臉部大小 40%
    clarity_score * 0.2        # 清晰度 20%
)
```

**Face Tracking / 臉部追蹤：**
- 使用 cosine similarity 比對 embeddings
- Threshold: 0.4 (可調整)
- 跨 frame 追蹤同一人物

**Identity Clustering / 身份聚類：**
```python
# 比對兩個臉是否為同一人
similarity = np.dot(emb1_norm, emb2_norm)
if similarity > threshold:
    same_person = True
```

**Output / 輸出：**
- Dominant face ID
- Appearance count
- Average face size
- Average clarity score
- Representative face crops

---

### 4. Face Swapper (face_swapper.py)

**Purpose / 目的：**
使用深度學習模型進行高品質臉部交換

**Model / 模型：**
- **Name**: inswapper_128.onnx
- **Input**: 128x128 RGB face images
- **Output**: Swapped face with same size
- **Source**: HuggingFace (deepinsight/inswapper)

**Swapping Process / 換臉流程：**

```
1. Extract source face region
   └──> Detect face bbox + landmarks
   
2. Align and crop face
   └──> Affine transformation to 128x128
   
3. Extract target face embedding
   └──> 512-dim normed embedding
   
4. Swap using inswapper model
   └──> source_face + target_embedding → swapped_face
   
5. Paste back to original frame
   └──> Inverse affine transformation + blending
```

**Face Tracking Feature / Face Tracking 功能：**
```python
# 當偵測失敗時使用前一個成功的位置
if current_face_detected:
    last_known_face = current_face
    tracking_counter = 0
elif tracking_counter < max_tracking_frames:
    use_face = last_known_face
    tracking_counter += 1
else:
    skip_frame = True
```

**Parameters / 參數：**
- `max_tracking_frames`: 5 frames
- `blend_ratio`: 0.75
- `paste_back`: True

---

### 5. Video Processor (video_processor.py)

**Purpose / 目的：**
處理完整影片並重建輸出

**Processing Pipeline / 處理管道：**

```
1. Read video metadata
   └──> FPS, resolution, total frames, duration
   
2. Process each frame
   ├──> Detect faces
   ├──> Match to dominant identity
   ├──> Swap matched faces
   └──> Save to temporary directory
   
3. Reconstruct video (FFmpeg)
   ├──> Frames → MP4 (H.264)
   ├──> Extract audio from original
   └──> Merge video + audio
```

**GPU Utilization / GPU 使用：**
```python
# 自動偵測可用 GPU
available_gpus = select_best_gpus()
# 當前使用單個 GPU (GPU 0)
# 支援 4 張 NVIDIA RTX PRO 6000
```

**FFmpeg Commands / FFmpeg 指令：**

```bash
# 1. Frames → Video
ffmpeg -framerate {fps} -i frame_%08d.png \
       -c:v libx264 -preset slow -crf 18 \
       -pix_fmt yuv420p temp_video.mp4

# 2. Extract Audio
ffmpeg -i input.mp4 -vn -acodec copy audio.aac

# 3. Merge Video + Audio
ffmpeg -i temp_video.mp4 -i audio.aac \
       -c:v copy -c:a aac output.mp4
```

**Performance / 效能：**
- 處理速度：~0.17 秒/frame (GPU)
- 8秒影片：約 30-35 秒
- 2小時影片：約 3-3.5 小時

---

## Data Flow / 資料流程

```
Input Video (MP4)
       ↓
[Frame Sampling]
   → Sampled Frames (NumPy arrays)
       ↓
[Face Detection]
   → Detected Faces (bbox, landmarks, embeddings)
       ↓
[Dominant Analysis]
   → Dominant Identity (avg_embedding, face_id)
       ↓
[Target Face Prep]
   → Target Faces (embeddings from images)
       ↓
[Face Swapping]
   → Processed Frames (NumPy arrays)
       ↓
[Video Reconstruction]
   → Output Video (MP4 with audio)
```

---

## Key Algorithms / 核心演算法

### 1. Cosine Similarity for Face Matching

```python
def cosine_similarity(emb1, emb2):
    """
    計算兩個 embedding 的相似度
    
    Args:
        emb1, emb2: 512-dim face embeddings
        
    Returns:
        similarity: float in [-1, 1]
                   closer to 1 = more similar
    """
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1_norm, emb2_norm)
```

### 2. Dominant Face Scoring

```python
def calculate_score(identity):
    """
    計算臉部的 dominance score
    
    Components:
    - Frequency: appearances / max_appearances
    - Size: avg_face_size / max_face_size  
    - Clarity: avg_confidence / max_confidence
    
    Returns:
        score: float in [0, 1]
    """
    freq_score = identity.appearances / max_appearances
    size_score = identity.avg_size / max_size
    clarity_score = identity.avg_confidence / max_confidence
    
    return (
        freq_score * 0.4 +
        size_score * 0.4 +
        clarity_score * 0.2
    )
```

### 3. Face Tracking Logic

```python
def track_face(current_faces, last_face, frames_since_detect):
    """
    Face tracking 邏輯
    
    當無法偵測到臉時，使用前一個已知位置
    最多追蹤 5 frames
    """
    if len(current_faces) > 0:
        # 成功偵測
        return current_faces[0], 0
    elif frames_since_detect < 5 and last_face is not None:
        # 使用 tracking
        return last_face, frames_since_detect + 1
    else:
        # 追蹤失敗
        return None, frames_since_detect + 1
```

---

## Configuration Reference / 配置參考

### config.yaml Structure

```yaml
# Frame Sampling
frame_sampling:
  target_samples: 200
  min_interval_seconds: 1.5
  save_debug_images: true

# Dominant Face Detection  
dominant_face:
  weight_frequency: 0.4
  weight_size: 0.4
  weight_clarity: 0.2
  min_face_size: 50
  detection_threshold: 0.3

# Face Processing
face_processing:
  model: inswapper_128
  face_enhancer: gfpgan_1.4
  blend_ratio: 0.75
  face_mask_blur: 0.3
  face_mask_padding: [0, 0, 0, 0]

# Video Processing
video:
  keep_fps: true
  keep_audio: true
  output_quality: high
  batch_size: 4
  temp_frame_format: png

# Hardware
hardware:
  execution_provider: cuda
  gpu_id: 0

# Debug
debug:
  verbose: true
  save_intermediate: true
  step_by_step_mode: true
```

---

## Performance Optimization / 效能優化

### Current Optimizations / 當前優化

1. **GPU Acceleration**
   - ONNX Runtime with CUDA
   - 10-20x faster than CPU

2. **Smart Sampling**
   - 分析 200 frames 而非全部
   - 節省 90%+ 分析時間

3. **Face Tracking**
   - 減少重複偵測
   - 提升 frame 覆蓋率 3-6%

4. **Batch Processing**
   - 可配置 batch_size
   - 減少 I/O overhead

### Future Improvements / 未來改進

1. **Multi-GPU Support**
   - 分段處理長影片
   - 理論可達 4x speedup

2. **Model Quantization**
   - INT8 quantization
   - 預期 2x faster

3. **Caching System**
   - Cache dominant face analysis
   - 重複處理同一影片更快

4. **Async I/O**
   - Async frame reading/writing
   - 減少 I/O wait time

---

## Error Handling / 錯誤處理

### Common Errors and Solutions

#### 1. CUDA Out of Memory
```python
try:
    result = swapper.swap(frame, face, target)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Fallback to CPU
        logger.warning("GPU OOM, falling back to CPU")
        use_cpu()
```

#### 2. No Face Detected
```python
if len(faces) == 0:
    if enable_tracking and last_face:
        # Use tracking
        return last_face
    else:
        # Skip frame
        return None
```

#### 3. CUDA Error (No Kernel)
```python
# Solution: Install compatible onnxruntime version
pip install onnxruntime-gpu==1.19.2
```

---

## Testing / 測試

### Unit Tests
```bash
# Run unit tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_face_detector.py
```

### Integration Tests
```bash
# Test full pipeline
python main.py --source tests/data/test_video.mp4 \
               --target tests/data/test_faces/ \
               --output tests/output/result.mp4 \
               --auto
```

### Performance Benchmarks
```bash
# Benchmark GPU vs CPU
python benchmark.py --mode gpu
python benchmark.py --mode cpu
```

---

## API Reference / API 參考

### Main Classes

#### FaceDetector
```python
detector = FaceDetector(provider='cuda')
detector.initialize()
faces = detector.detect_faces(frame)
```

#### DominantAnalyzer
```python
analyzer = DominantAnalyzer(config)
dominant = analyzer.analyze(frame_faces)
```

#### FaceSwapper
```python
swapper = FaceSwapper(provider='cuda')
result = swapper.swap_faces_in_frame(frame, source_faces, target_faces)
```

#### VideoProcessor
```python
processor = VideoProcessor(detector, swapper, config)
stats = processor.process_video(input_path, output_path, dominant, targets)
```

---

## Troubleshooting Guide / 故障排除指南

### Debug Mode

啟用 verbose logging：
```bash
python main.py --source input.mp4 \
               --target faces/ \
               --output output.mp4 \
               --verbose
```

### Check GPU Usage
```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Profile Performance
```python
import cProfile
cProfile.run('main()', 'profile.stats')

# Analyze stats
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

---

## Contributing / 貢獻

### Code Style
- Follow PEP 8
- Use type hints
- Document all functions

### Testing Requirements
- All new features must have tests
- Maintain >80% code coverage

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit PR with description

---

## License / 授權

本專案僅供研究與學習使用。

使用的開源組件：
- InsightFace: MIT License
- GFPGAN: Apache 2.0 License
- ONNX Runtime: MIT License

---

## References / 參考資料

### Papers
1. ArcFace: Additive Angular Margin Loss (Deng et al., 2019)
2. InsightFace: 2D and 3D Face Analysis Project
3. GFPGAN: Towards Real-World Blind Face Restoration

### Resources
- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [ONNX Runtime Docs](https://onnxruntime.ai/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
