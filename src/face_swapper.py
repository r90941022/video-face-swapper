"""
Face Swapper Module

High-quality face swapping using InsightFace's inswapper model
with optional face enhancement using GFPGAN
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path
import onnxruntime

from face_detector import DetectedFace

logger = logging.getLogger(__name__)


class FaceSwapper:
    """Face swapper using InsightFace inswapper model"""

    def __init__(self,
                 model_path: str = None,
                 use_enhancer: bool = True,
                 enhancer_model: str = 'gfpgan_1.4',
                 blend_ratio: float = 0.75,
                 face_mask_blur: float = 0.3,
                 providers: List[str] = None):
        """
        Initialize face swapper

        Args:
            model_path: Path to inswapper model (will auto-download if None)
            use_enhancer: Whether to use face enhancement
            enhancer_model: Face enhancer model name
            blend_ratio: Blending ratio for face fusion
            face_mask_blur: Face mask edge blur amount
            providers: ONNX Runtime providers
        """
        self.model_path = model_path
        self.use_enhancer = use_enhancer
        self.enhancer_model = enhancer_model
        self.blend_ratio = blend_ratio
        self.face_mask_blur = face_mask_blur
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.swapper = None
        self.enhancer = None

    def initialize(self):
        """Initialize face swapper and enhancer models"""
        logger.info("Initializing face swapper...")

        # Initialize InsightFace swapper
        self._init_swapper()

        # Initialize enhancer if requested
        if self.use_enhancer:
            self._init_enhancer()

        logger.info("Face swapper initialized successfully")

    def _init_swapper(self):
        """Initialize InsightFace inswapper model"""
        try:
            import insightface
            from insightface.model_zoo import get_model

            # Download/load inswapper model
            if self.model_path is None:
                # Use default path and auto-download
                model_name = 'inswapper_128.onnx'
                model_dir = Path.home() / '.insightface' / 'models'
                model_dir.mkdir(parents=True, exist_ok=True)
                self.model_path = str(model_dir / model_name)

                # Check if model exists, if not download
                if not Path(self.model_path).exists():
                    logger.info(f"Downloading inswapper model to {self.model_path}...")
                    # InsightFace will auto-download when we call get_model

            # Load model
            self.swapper = insightface.model_zoo.get_model(
                self.model_path,
                download=True,
                download_zip=True,
                providers=self.providers
            )

            logger.info(f"Loaded inswapper model from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize swapper: {e}")
            logger.info("Attempting alternative initialization method...")

            # Try alternative: direct ONNX loading
            try:
                from insightface.utils import face_align
                import onnxruntime as ort

                # This is a fallback - we'll implement basic swapping
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                try:
                    self.swapper = ort.InferenceSession(
                        self.model_path,
                        sess_options=session_options,
                        providers=self.providers
                    )
                    logger.info("Loaded model using ONNX Runtime directly")
                except Exception as cuda_err:
                    if 'CUDA' in str(cuda_err) or 'cudaErrorNoKernelImageForDevice' in str(cuda_err):
                        logger.warning(f"CUDA failed for swapper: {cuda_err}")
                        logger.warning("Falling back to CPU for face swapper...")
                        self.swapper = ort.InferenceSession(
                            self.model_path,
                            sess_options=session_options,
                            providers=['CPUExecutionProvider']
                        )
                        logger.info("Loaded model using CPU provider")
                    else:
                        raise

            except Exception as e2:
                logger.error(f"Failed alternative initialization: {e2}")
                raise RuntimeError("Could not initialize face swapper model")

    def _init_enhancer(self):
        """Initialize GFPGAN face enhancer"""
        try:
            from gfpgan import GFPGANer

            logger.info(f"Initializing face enhancer: {self.enhancer_model}")

            # Determine model path
            if self.enhancer_model == 'gfpgan_1.4':
                model_name = 'GFPGANv1.4.pth'
            elif self.enhancer_model == 'gfpgan_1.3':
                model_name = 'GFPGANv1.3.pth'
            else:
                model_name = 'GFPGANv1.4.pth'

            model_dir = Path.home() / '.cache' / 'gfpgan' / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_name

            self.enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=1,  # No upscaling, just enhancement
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device='cuda' if 'CUDAExecutionProvider' in self.providers else 'cpu'
            )

            logger.info("Face enhancer initialized")

        except ImportError:
            logger.warning("GFPGAN not installed, face enhancement disabled")
            self.use_enhancer = False
        except Exception as e:
            logger.warning(f"Failed to initialize enhancer: {e}")
            self.use_enhancer = False

    def prepare_target_faces(self,
                           target_image_paths: List[str],
                           detector=None) -> List[DetectedFace]:
        """
        Prepare target faces from images

        Args:
            target_image_paths: List of paths to target face images
            detector: FaceDetector instance (will create if None)

        Returns:
            List of detected target faces
        """
        if detector is None:
            from face_detector import FaceDetector
            detector = FaceDetector()
            detector.initialize()

        logger.info(f"Preparing {len(target_image_paths)} target faces...")

        target_faces = []
        for img_path in target_image_paths:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue

            faces = detector.detect_faces(img)
            if len(faces) == 0:
                logger.warning(f"No face detected in: {img_path}")
                continue

            # Use the largest face
            largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            target_faces.append(largest_face)

            logger.info(f"  ✓ Extracted face from {Path(img_path).name}")

        logger.info(f"Prepared {len(target_faces)} target faces")
        return target_faces

    def swap_face(self,
                  frame: np.ndarray,
                  source_face: DetectedFace,
                  target_face: DetectedFace) -> np.ndarray:
        """
        Swap a single face in frame

        Args:
            frame: Input frame (BGR)
            source_face: Face to replace in frame
            target_face: Face to swap in

        Returns:
            Frame with swapped face
        """
        if self.swapper is None:
            self.initialize()

        try:
            # Convert DetectedFace to InsightFace face object format
            # InsightFace swapper expects a face object with 'kps' attribute
            import types
            source_face_obj = types.SimpleNamespace()
            source_face_obj.bbox = source_face.bbox
            source_face_obj.kps = source_face.landmarks  # InsightFace uses 'kps' not 'landmarks'
            source_face_obj.det_score = source_face.confidence

            # Also prepare target face object with normed_embedding
            target_face_obj = types.SimpleNamespace()
            target_face_obj.embedding = target_face.embedding
            # Normalize embedding
            import numpy as np
            target_face_obj.normed_embedding = target_face.embedding / np.linalg.norm(target_face.embedding)

            # Use InsightFace swapper
            result = self.swapper.get(
                frame,
                source_face_obj,
                target_face_obj,
                paste_back=True
            )

            # Apply face enhancement if enabled
            if self.use_enhancer and self.enhancer is not None:
                result = self._enhance_face(result, source_face)

            return result

        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            return frame  # Return original frame on error

    def swap_faces_in_frame(self,
                           frame: np.ndarray,
                           faces_to_swap: List[DetectedFace],
                           target_faces: List[DetectedFace]) -> np.ndarray:
        """
        Swap multiple faces in a frame

        Args:
            frame: Input frame
            faces_to_swap: List of faces to replace
            target_faces: List of target faces (will use best match or first)

        Returns:
            Frame with swapped faces
        """
        result = frame.copy()

        for source_face in faces_to_swap:
            # Select best matching target face
            # For now, just use the first target face
            # TODO: Could implement angle/pose matching
            target_face = target_faces[0] if target_faces else None

            if target_face is not None:
                result = self.swap_face(result, source_face, target_face)

        return result

    def _enhance_face(self,
                     frame: np.ndarray,
                     face: DetectedFace) -> np.ndarray:
        """
        Enhance face region using GFPGAN

        Args:
            frame: Frame with swapped face
            face: Face region to enhance

        Returns:
            Enhanced frame
        """
        if self.enhancer is None:
            return frame

        try:
            # Get face bbox with some padding
            bbox = face.bbox
            x1, y1, x2, y2 = bbox

            # Add padding
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            # Extract face region
            face_region = frame[y1:y2, x1:x2]

            # Enhance
            _, _, enhanced = self.enhancer.enhance(
                face_region,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )

            # Put enhanced region back
            if enhanced is not None:
                frame[y1:y2, x1:x2] = enhanced

            return frame

        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return frame

    def create_seamless_mask(self,
                            frame_shape: Tuple[int, int],
                            face: DetectedFace,
                            blur_amount: float = None) -> np.ndarray:
        """
        Create a seamless blending mask for face region

        Args:
            frame_shape: (height, width) of frame
            face: Detected face
            blur_amount: Blur kernel size ratio (None = use default)

        Returns:
            Blending mask (0-1 float)
        """
        if blur_amount is None:
            blur_amount = self.face_mask_blur

        mask = np.zeros(frame_shape[:2], dtype=np.float32)

        # Create mask from face landmarks or bbox
        if face.landmarks is not None and len(face.landmarks) > 5:
            # Use convex hull of landmarks
            points = face.landmarks.astype(np.int32)
            cv2.fillConvexPoly(mask, points, 1.0)
        else:
            # Use bbox
            bbox = face.bbox.astype(int)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0

        # Apply Gaussian blur for seamless blending
        kernel_size = int(min(frame_shape[:2]) * blur_amount)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        return mask


class FaceSwapperV2:
    """
    Alternative face swapper implementation with more control
    Uses direct ONNX Runtime for finer control
    """

    def __init__(self, model_path: str, providers: List[str] = None):
        self.model_path = model_path
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = None

    def initialize(self):
        """Initialize ONNX session"""
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=self.providers
        )

        logger.info(f"Initialized FaceSwapperV2 with {self.session.get_providers()}")

    def swap(self,
             source_face_embedding: np.ndarray,
             target_face_image: np.ndarray) -> np.ndarray:
        """
        Perform face swap

        Args:
            source_face_embedding: 512-dim embedding of target identity
            target_face_image: Aligned face image to modify

        Returns:
            Swapped face image
        """
        if self.session is None:
            self.initialize()

        # Prepare inputs based on model requirements
        # This is model-specific and may need adjustment
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        # Run inference
        result = self.session.run(
            [output_name],
            {input_name: target_face_image}
        )[0]

        return result


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use main.py to run the full pipeline.")
