"""
Face Detection Module

Uses InsightFace for high-quality face detection and feature extraction
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Container for detected face information"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # 5 or 106 facial landmarks
    confidence: float
    embedding: Optional[np.ndarray] = None  # 512-dim feature vector
    frame_idx: Optional[int] = None
    face_id: Optional[int] = None  # Unique ID assigned during tracking
    gender: Optional[str] = None  # 'M' for male, 'F' for female
    age: Optional[int] = None  # Estimated age


class FaceDetector:
    """Face detector using InsightFace"""

    def __init__(self,
                 detection_threshold: float = 0.5,
                 min_face_size: int = 80,
                 provider: str = 'cuda'):
        """
        Initialize face detector

        Args:
            detection_threshold: Minimum confidence for face detection
            min_face_size: Minimum face size in pixels
            provider: Execution provider ('cuda', 'cpu', 'coreml')
        """
        self.detection_threshold = detection_threshold
        self.min_face_size = min_face_size
        self.provider = provider
        self.app = None

    def initialize(self):
        """Initialize InsightFace models"""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info("Initializing InsightFace...")

            # Determine providers based on user preference
            providers = self._get_providers()

            # Initialize face analysis
            try:
                self.app = FaceAnalysis(
                    name='buffalo_l',
                    providers=providers
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info(f"InsightFace initialized with providers: {providers}")
            except Exception as cuda_error:
                if 'CUDA' in str(cuda_error) or 'cudaErrorNoKernelImageForDevice' in str(cuda_error):
                    logger.warning(f"CUDA initialization failed: {cuda_error}")
                    logger.warning("Falling back to CPU...")
                    self.app = FaceAnalysis(
                        name='buffalo_l',
                        providers=['CPUExecutionProvider']
                    )
                    self.app.prepare(ctx_id=-1, det_size=(640, 640))
                    logger.info("InsightFace initialized with CPU provider")
                else:
                    raise

        except ImportError:
            raise ImportError(
                "InsightFace not installed. Please install: pip install insightface"
            )
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime providers based on hardware"""
        if self.provider == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.provider == 'coreml':
            return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detected faces
        """
        if self.app is None:
            self.initialize()

        # Detect faces with CUDA error handling
        try:
            faces = self.app.get(frame)
        except Exception as e:
            if 'CUDA' in str(e) or 'cudaErrorNoKernelImageForDevice' in str(e):
                logger.warning(f"CUDA error during inference: {e}")
                logger.warning("Reinitializing with CPU provider...")
                # Reinitialize with CPU
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
                logger.info("Reinitialized with CPU, retrying detection...")
                faces = self.app.get(frame)
            else:
                raise

        # Convert to DetectedFace objects
        detected_faces = []
        for face in faces:
            # Filter by confidence and size
            if face.det_score < self.detection_threshold:
                continue

            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]

            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue

            # Extract gender and age if available
            gender = None
            age = None
            if hasattr(face, 'gender'):
                # InsightFace returns 0 for female, 1 for male
                gender = 'F' if face.gender == 0 else 'M'
            if hasattr(face, 'age'):
                age = int(face.age)

            detected_face = DetectedFace(
                bbox=bbox,
                landmarks=face.kps,
                confidence=float(face.det_score),
                embedding=face.embedding if hasattr(face, 'embedding') else None,
                gender=gender,
                age=age
            )
            detected_faces.append(detected_face)

        return detected_faces

    def detect_faces_batch(self,
                          frames: List[np.ndarray],
                          frame_indices: List[int] = None,
                          save_debug: bool = False,
                          output_dir: str = None) -> Dict[int, List[DetectedFace]]:
        """
        Detect faces in multiple frames

        Args:
            frames: List of frames
            frame_indices: List of frame indices (optional)
            save_debug: Whether to save debug visualizations
            output_dir: Output directory for debug images

        Returns:
            Dictionary mapping frame_idx -> list of DetectedFace
        """
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        logger.info(f"Detecting faces in {len(frames)} frames...")

        results = {}

        from tqdm import tqdm
        for frame, frame_idx in tqdm(zip(frames, frame_indices),
                                     total=len(frames),
                                     desc="Detecting faces"):
            faces = self.detect_faces(frame)

            # Assign frame index to each face
            for face in faces:
                face.frame_idx = frame_idx

            results[frame_idx] = faces

            # Save debug visualization
            if save_debug and output_dir:
                self._save_debug_frame(frame, faces, frame_idx, output_dir)

        # Log statistics
        total_faces = sum(len(faces) for faces in results.values())
        frames_with_faces = sum(1 for faces in results.values() if len(faces) > 0)

        logger.info(f"Detection complete:")
        logger.info(f"  - Total faces detected: {total_faces}")
        logger.info(f"  - Frames with faces: {frames_with_faces}/{len(frames)}")
        logger.info(f"  - Average faces per frame: {total_faces/len(frames):.2f}")

        return results

    def _save_debug_frame(self,
                         frame: np.ndarray,
                         faces: List[DetectedFace],
                         frame_idx: int,
                         output_dir: str):
        """Save frame with face bounding boxes drawn"""
        output_path = Path(output_dir) / "detected_faces"
        output_path.mkdir(parents=True, exist_ok=True)

        # Draw bounding boxes and landmarks
        debug_frame = frame.copy()

        for idx, face in enumerate(faces):
            # Draw bounding box
            bbox = face.bbox
            cv2.rectangle(debug_frame,
                         (bbox[0], bbox[1]),
                         (bbox[2], bbox[3]),
                         (0, 255, 0), 2)

            # Draw confidence
            text = f"{face.confidence:.2f}"
            cv2.putText(debug_frame, text,
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw landmarks
            if face.landmarks is not None:
                for point in face.landmarks:
                    cv2.circle(debug_frame, (int(point[0]), int(point[1])),
                             2, (0, 0, 255), -1)

        # Save
        filename = f"frame_{frame_idx:06d}_faces{len(faces)}.jpg"
        cv2.imwrite(str(output_path / filename), debug_frame)

    def extract_face_crop(self,
                         frame: np.ndarray,
                         face: DetectedFace,
                         padding: float = 0.3) -> np.ndarray:
        """
        Extract face crop from frame with padding

        Args:
            frame: Input frame
            face: Detected face
            padding: Padding ratio around face bbox

        Returns:
            Cropped face image
        """
        bbox = face.bbox
        x1, y1, x2, y2 = bbox

        # Add padding
        width = x2 - x1
        height = y2 - y1
        pad_w = int(width * padding)
        pad_h = int(height * padding)

        # Expand bbox with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(frame.shape[1], x2 + pad_w)
        y2 = min(frame.shape[0], y2 + pad_h)

        # Crop
        face_crop = frame[y1:y2, x1:x2]

        return face_crop

    def compute_face_similarity(self,
                               face1: DetectedFace,
                               face2: DetectedFace) -> float:
        """
        Compute cosine similarity between two face embeddings

        Args:
            face1: First face
            face2: Second face

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if face1.embedding is None or face2.embedding is None:
            return 0.0

        # Normalize embeddings
        emb1 = face1.embedding / np.linalg.norm(face1.embedding)
        emb2 = face2.embedding / np.linalg.norm(face2.embedding)

        # Cosine similarity
        similarity = np.dot(emb1, emb2)

        return float(similarity)


def test_face_detector():
    """Test function for face detector"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_detector.py <image_or_video_path>")
        return

    input_path = sys.argv[1]
    output_dir = "/tmp/video-changer/output"

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create detector
    detector = FaceDetector(detection_threshold=0.5,
                           min_face_size=80,
                           provider='cuda')

    # Test on image or video
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Single image
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Cannot read image: {input_path}")
            return

        faces = detector.detect_faces(frame)
        print(f"\nDetected {len(faces)} faces")

        for idx, face in enumerate(faces):
            print(f"Face {idx+1}:")
            print(f"  - Confidence: {face.confidence:.3f}")
            print(f"  - Bbox: {face.bbox}")
            print(f"  - Embedding shape: {face.embedding.shape if face.embedding is not None else None}")

    else:
        # Video - test with frame sampler
        from frame_sampler import FrameSampler

        sampler = FrameSampler(target_samples=50, save_debug=False)
        frames, indices, metadata = sampler.sample_frames(input_path)

        results = detector.detect_faces_batch(frames, indices,
                                              save_debug=True,
                                              output_dir=output_dir)

        print(f"\nCheck output directory: {output_dir}/detected_faces")


if __name__ == "__main__":
    test_face_detector()
