"""
Video Processing Pipeline

Handles full video processing: reading, face swapping, and writing output
"""

import cv2
import numpy as np
from typing import List, Optional, Dict
import logging
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from face_detector import DetectedFace, FaceDetector
from face_swapper import FaceSwapper
from dominant_analyzer import FaceIdentity
from gpu_utils import select_best_gpus

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process full videos with face swapping"""

    def __init__(self,
                 detector: FaceDetector,
                 swapper: FaceSwapper,
                 keep_fps: bool = True,
                 keep_audio: bool = True,
                 output_quality: str = 'high',
                 batch_size: int = 4,
                 use_multi_gpu: bool = True):
        """
        Initialize video processor

        Args:
            detector: Face detector instance
            swapper: Face swapper instance
            keep_fps: Preserve original FPS
            keep_audio: Preserve audio track
            output_quality: Output quality ('low', 'medium', 'high')
            batch_size: Number of frames to process in parallel
            use_multi_gpu: Enable multi-GPU processing
        """
        self.detector = detector
        self.swapper = swapper
        self.keep_fps = keep_fps
        self.keep_audio = keep_audio
        self.output_quality = output_quality
        self.batch_size = batch_size
        self.use_multi_gpu = use_multi_gpu

        # Detect available GPUs
        if use_multi_gpu:
            self.available_gpus = select_best_gpus()
            if len(self.available_gpus) > 1:
                logger.info(f"Multi-GPU mode enabled: using {len(self.available_gpus)} GPUs {self.available_gpus}")
            elif len(self.available_gpus) == 1:
                logger.info(f"Single GPU mode: using GPU {self.available_gpus[0]}")
            else:
                logger.warning("No GPUs detected, falling back to CPU")
        else:
            self.available_gpus = []
            logger.info("Multi-GPU mode disabled")

    def process_video(self,
                     input_video_path: str,
                     output_video_path: str,
                     dominant_identity: FaceIdentity,
                     target_faces: List[DetectedFace],
                     progress_callback=None) -> Dict:
        """
        Process entire video with face swapping

        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            dominant_identity: The dominant face identity to replace
            target_faces: Target faces to swap in
            progress_callback: Optional callback function(current, total)

        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing video: {input_video_path}")
        logger.info(f"Output will be saved to: {output_video_path}")

        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video properties:")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - FPS: {fps}")
        logger.info(f"  - Resolution: {width}x{height}")

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Using temporary directory: {temp_path}")

            # Process frames
            stats = self._process_frames(
                cap, temp_path, dominant_identity, target_faces,
                total_frames, progress_callback
            )

            cap.release()

            # Reconstruct video
            logger.info("Reconstructing video...")
            self._reconstruct_video(
                temp_path, input_video_path, output_video_path,
                width, height, fps, total_frames
            )

        logger.info("Video processing complete!")
        return stats

    def _process_frames(self,
                       cap: cv2.VideoCapture,
                       output_dir: Path,
                       dominant_identity: FaceIdentity,
                       target_faces: List[DetectedFace],
                       total_frames: int,
                       progress_callback=None) -> Dict:
        """
        Process all frames and save to temporary directory

        Returns:
            Statistics dictionary
        """
        # Multi-GPU processing disabled due to ONNX Runtime limitations
        # Use single GPU processing for stability
        if False and len(self.available_gpus) > 1 and total_frames > 100:
            logger.info(f"Using multi-GPU processing with {len(self.available_gpus)} GPUs")
            return self._process_frames_multi_gpu(
                cap, output_dir, dominant_identity, target_faces,
                total_frames, progress_callback
            )

        # Single GPU or CPU processing
        frame_idx = 0
        swapped_count = 0
        skipped_count = 0
        error_count = 0
        tracked_count = 0

        # Get dominant face embedding for matching
        dominant_embedding = dominant_identity.avg_embedding

        # Face tracking variables
        last_detected_face = None
        frames_since_detection = 0
        max_tracking_frames = 5  # Maximum frames to track without detection

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Detect faces in current frame
                    faces = self.detector.detect_faces(frame)

                    # Find faces matching dominant identity
                    faces_to_swap = self._find_matching_faces(
                        faces, dominant_embedding, threshold=0.4
                    )

                    if len(faces_to_swap) > 0:
                        # Swap faces
                        result_frame = self.swapper.swap_faces_in_frame(
                            frame, faces_to_swap, target_faces
                        )
                        swapped_count += 1
                        # Update tracking
                        last_detected_face = faces_to_swap[0]
                        frames_since_detection = 0
                    elif last_detected_face is not None and frames_since_detection < max_tracking_frames:
                        # Use face tracking - try to swap using last known position
                        logger.debug(f"Frame {frame_idx}: Using face tracking (no detection)")
                        try:
                            result_frame = self.swapper.swap_faces_in_frame(
                                frame, [last_detected_face], target_faces
                            )
                            swapped_count += 1
                            tracked_count += 1
                            frames_since_detection += 1
                        except:
                            # Tracking failed, keep original
                            result_frame = frame
                            skipped_count += 1
                            frames_since_detection += 1
                    else:
                        # No matching face and no tracking available, keep original
                        result_frame = frame
                        skipped_count += 1
                        frames_since_detection += 1
                        # Reset tracking if too many frames without detection
                        if frames_since_detection > max_tracking_frames:
                            last_detected_face = None

                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    result_frame = frame
                    error_count += 1

                # Save frame
                frame_filename = output_dir / f"frame_{frame_idx:08d}.png"
                cv2.imwrite(str(frame_filename), result_frame)

                frame_idx += 1
                pbar.update(1)

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        stats = {
            'total_frames': frame_idx,
            'swapped_frames': swapped_count,
            'skipped_frames': skipped_count,
            'error_frames': error_count,
            'tracked_frames': tracked_count
        }

        logger.info(f"Processing statistics:")
        logger.info(f"  - Total frames: {stats['total_frames']}")
        logger.info(f"  - Swapped: {stats['swapped_frames']}")
        logger.info(f"  - Tracked: {stats.get('tracked_frames', 0)}")
        logger.info(f"  - Skipped: {stats['skipped_frames']}")
        logger.info(f"  - Errors: {stats['error_frames']}")

        return stats

    def _find_matching_faces(self,
                           faces: List[DetectedFace],
                           reference_embedding: np.ndarray,
                           threshold: float = 0.4) -> List[DetectedFace]:
        """
        Find faces matching the reference embedding

        Args:
            faces: List of detected faces
            reference_embedding: Reference embedding to match
            threshold: Similarity threshold

        Returns:
            List of matching faces
        """
        matching_faces = []

        if reference_embedding is None:
            return matching_faces

        for face in faces:
            if face.embedding is None:
                continue

            # Compute similarity
            face_emb = face.embedding / np.linalg.norm(face.embedding)
            ref_emb = reference_embedding / np.linalg.norm(reference_embedding)
            similarity = np.dot(face_emb, ref_emb)

            if similarity > threshold:
                matching_faces.append(face)

        return matching_faces

    def _reconstruct_video(self,
                          frames_dir: Path,
                          input_video: str,
                          output_video: str,
                          width: int,
                          height: int,
                          fps: float,
                          total_frames: int):
        """
        Reconstruct video from processed frames using ffmpeg

        Args:
            frames_dir: Directory containing processed frames
            input_video: Original video path (for audio extraction)
            output_video: Output video path
            width: Video width
            height: Video height
            fps: Frames per second
            total_frames: Total number of frames
        """
        # Ensure output directory exists
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine quality settings
        if self.output_quality == 'high':
            crf = '18'
            preset = 'slow'
        elif self.output_quality == 'medium':
            crf = '23'
            preset = 'medium'
        else:  # low
            crf = '28'
            preset = 'fast'

        # Create video from frames
        frames_pattern = str(frames_dir / "frame_%08d.png")
        temp_video = str(frames_dir / "temp_video.mp4")

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', frames_pattern,
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', crf,
            '-pix_fmt', 'yuv420p',
            temp_video
        ]

        logger.info("Running ffmpeg to create video...")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr.decode()}")
            raise RuntimeError("Failed to create video from frames")

        # Add audio if requested
        if self.keep_audio:
            logger.info("Adding audio track...")
            self._add_audio(temp_video, input_video, output_video)
        else:
            # Just move the temp video to output
            shutil.move(temp_video, output_video)

        logger.info(f"Video saved to: {output_video}")

    def _add_audio(self,
                   video_file: str,
                   source_video: str,
                   output_file: str):
        """
        Add audio from source video to output video

        Args:
            video_file: Video file without audio
            source_video: Source video with audio
            output_file: Output video with audio
        """
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_file,
            '-i', source_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',  # '?' makes audio optional
            '-shortest',
            output_file
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to add audio: {e.stderr.decode()}")
            logger.warning("Saving video without audio")
            shutil.copy(video_file, output_file)

    def preview_swap(self,
                    frame: np.ndarray,
                    dominant_identity: FaceIdentity,
                    target_faces: List[DetectedFace]) -> np.ndarray:
        """
        Create a preview of face swap on a single frame

        Args:
            frame: Input frame
            dominant_identity: Dominant face identity
            target_faces: Target faces

        Returns:
            Frame with swapped face
        """
        # Detect faces
        faces = self.detector.detect_faces(frame)

        # Find matching faces
        faces_to_swap = self._find_matching_faces(
            faces, dominant_identity.avg_embedding, threshold=0.4
        )

        if len(faces_to_swap) > 0:
            result = self.swapper.swap_faces_in_frame(
                frame, faces_to_swap, target_faces
            )
        else:
            result = frame.copy()
            # Draw message
            cv2.putText(result, "No dominant face found in this frame",
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return result

    def create_preview_comparison(self,
                                 frames: List[np.ndarray],
                                 frame_indices: List[int],
                                 dominant_identity: FaceIdentity,
                                 target_faces: List[DetectedFace],
                                 output_path: str,
                                 num_preview: int = 10):
        """
        Create side-by-side comparison of original vs swapped frames

        Args:
            frames: List of frames
            frame_indices: Frame indices
            dominant_identity: Dominant face identity
            target_faces: Target faces
            output_path: Output path for comparison image
            num_preview: Number of frames to preview
        """
        logger.info(f"Creating preview comparison with {num_preview} frames...")

        # Sample frames for preview
        sample_indices = np.linspace(0, len(frames)-1, num_preview, dtype=int)

        comparison_pairs = []
        for idx in sample_indices:
            frame = frames[idx]
            swapped = self.preview_swap(frame, dominant_identity, target_faces)

            # Resize for display
            h, w = frame.shape[:2]
            display_w = 400
            display_h = int(h * display_w / w)

            original_resized = cv2.resize(frame, (display_w, display_h))
            swapped_resized = cv2.resize(swapped, (display_w, display_h))

            # Add labels
            cv2.putText(original_resized, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(swapped_resized, "Swapped", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Concatenate side by side
            pair = np.hstack([original_resized, swapped_resized])
            comparison_pairs.append(pair)

        # Stack vertically
        grid = np.vstack(comparison_pairs)

        # Save
        cv2.imwrite(output_path, grid)
        logger.info(f"Preview comparison saved to: {output_path}")

    def _process_frames_multi_gpu(self,
                                   cap: cv2.VideoCapture,
                                   output_dir: Path,
                                   dominant_identity: FaceIdentity,
                                   target_faces: List[DetectedFace],
                                   total_frames: int,
                                   progress_callback=None) -> Dict:
        """
        Process frames using multiple GPUs by cycling through them

        Returns:
            Statistics dictionary
        """
        num_gpus = len(self.available_gpus)
        logger.info(f"Using round-robin GPU scheduling across {num_gpus} GPUs")

        # Create detector and swapper instances for each GPU
        gpu_workers = []
        for gpu_id in self.available_gpus:
            logger.info(f"Initializing workers for GPU {gpu_id}...")
            # Set CUDA device
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            detector = FaceDetector(provider='cuda')
            detector.initialize()

            swapper = FaceSwapper(provider='cuda')
            swapper.initialize()

            gpu_workers.append({
                'gpu_id': gpu_id,
                'detector': detector,
                'swapper': swapper
            })

        # Reset to see all GPUs
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

        # Process frames in batches, cycling through GPUs
        frame_idx = 0
        swapped_count = 0
        skipped_count = 0
        error_count = 0
        tracked_count = 0

        dominant_embedding = dominant_identity.avg_embedding
        last_detected_face = None
        frames_since_detection = 0
        max_tracking_frames = 5

        gpu_cycle_idx = 0

        with tqdm(total=total_frames, desc="Processing frames (multi-GPU)") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Select GPU worker in round-robin fashion
                worker = gpu_workers[gpu_cycle_idx]
                gpu_cycle_idx = (gpu_cycle_idx + 1) % num_gpus

                try:
                    # Detect faces using current GPU
                    faces = worker['detector'].detect_faces(frame)

                    # Find matching faces
                    faces_to_swap = self._find_matching_faces(
                        faces, dominant_embedding, threshold=0.4
                    )

                    if len(faces_to_swap) > 0:
                        # Swap faces using current GPU
                        result_frame = worker['swapper'].swap_faces_in_frame(
                            frame, faces_to_swap, target_faces
                        )
                        swapped_count += 1
                        last_detected_face = faces_to_swap[0]
                        frames_since_detection = 0
                    elif last_detected_face is not None and frames_since_detection < max_tracking_frames:
                        # Use tracking
                        try:
                            result_frame = worker['swapper'].swap_faces_in_frame(
                                frame, [last_detected_face], target_faces
                            )
                            swapped_count += 1
                            tracked_count += 1
                            frames_since_detection += 1
                        except:
                            result_frame = frame
                            skipped_count += 1
                            frames_since_detection += 1
                    else:
                        result_frame = frame
                        skipped_count += 1
                        frames_since_detection += 1
                        if frames_since_detection > max_tracking_frames:
                            last_detected_face = None

                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    result_frame = frame
                    error_count += 1

                # Save frame
                frame_filename = output_dir / f"frame_{frame_idx:08d}.png"
                cv2.imwrite(str(frame_filename), result_frame)

                frame_idx += 1
                pbar.update(1)

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        stats = {
            'total_frames': frame_idx,
            'swapped_frames': swapped_count,
            'skipped_frames': skipped_count,
            'error_frames': error_count,
            'tracked_frames': tracked_count
        }

        logger.info(f"Multi-GPU processing complete:")
        logger.info(f"  - Total frames: {stats['total_frames']}")
        logger.info(f"  - Swapped: {stats['swapped_frames']}")
        logger.info(f"  - Tracked: {stats['tracked_frames']}")
        logger.info(f"  - Skipped: {stats['skipped_frames']}")
        logger.info(f"  - Errors: {stats['error_frames']}")

        return stats


def _process_chunk_worker(frames, start_idx, gpu_id, dominant_identity, target_faces, output_dir):
    """
    Worker function to process a chunk of frames on a specific GPU
    This runs in a separate process
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    from face_detector import FaceDetector
    from face_swapper import FaceSwapper
    import numpy as np
    import cv2

    # Initialize detector and swapper for this GPU
    detector = FaceDetector(provider='cuda')
    detector.initialize()

    swapper = FaceSwapper(provider='cuda')
    swapper.initialize()

    swapped_count = 0
    skipped_count = 0
    error_count = 0
    tracked_count = 0

    dominant_embedding = dominant_identity.avg_embedding
    last_detected_face = None
    frames_since_detection = 0
    max_tracking_frames = 5

    for i, frame in enumerate(frames):
        frame_idx = start_idx + i

        try:
            # Detect faces
            faces = detector.detect_faces(frame)

            # Find matching faces
            matching_faces = []
            for face in faces:
                if face.embedding is not None and dominant_embedding is not None:
                    face_emb = face.embedding / np.linalg.norm(face.embedding)
                    ref_emb = dominant_embedding / np.linalg.norm(dominant_embedding)
                    similarity = np.dot(face_emb, ref_emb)
                    if similarity > 0.4:
                        matching_faces.append(face)

            if len(matching_faces) > 0:
                # Swap faces
                result_frame = swapper.swap_faces_in_frame(frame, matching_faces, target_faces)
                swapped_count += 1
                last_detected_face = matching_faces[0]
                frames_since_detection = 0
            elif last_detected_face is not None and frames_since_detection < max_tracking_frames:
                # Use tracking
                try:
                    result_frame = swapper.swap_faces_in_frame(frame, [last_detected_face], target_faces)
                    swapped_count += 1
                    tracked_count += 1
                    frames_since_detection += 1
                except:
                    result_frame = frame
                    skipped_count += 1
                    frames_since_detection += 1
            else:
                result_frame = frame
                skipped_count += 1
                frames_since_detection += 1
                if frames_since_detection > max_tracking_frames:
                    last_detected_face = None

        except Exception as e:
            result_frame = frame
            error_count += 1

        # Save frame
        frame_filename = output_dir / f"frame_{frame_idx:08d}.png"
        cv2.imwrite(str(frame_filename), result_frame)

    return {
        'swapped': swapped_count,
        'skipped': skipped_count,
        'errors': error_count,
        'tracked': tracked_count
    }


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use main.py to run the full pipeline.")
