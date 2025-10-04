"""
Smart Frame Sampling Module

This module implements intelligent frame sampling strategy that:
- Samples frames uniformly across the entire video duration
- Adapts sampling interval based on video length
- Ensures sufficient samples (target: 200 frames) for robust analysis
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FrameSampler:
    """Smart frame sampler for video analysis"""

    def __init__(self,
                 target_samples: int = 200,
                 min_interval_seconds: float = 1.5,
                 save_debug: bool = True):
        """
        Initialize frame sampler

        Args:
            target_samples: Target number of frames to sample
            min_interval_seconds: Minimum interval between samples for long videos
            save_debug: Whether to save debug visualization
        """
        self.target_samples = target_samples
        self.min_interval_seconds = min_interval_seconds
        self.save_debug = save_debug

    def sample_frames(self,
                      video_path: str,
                      output_dir: str = None) -> Tuple[List[np.ndarray], List[int], Dict]:
        """
        Sample frames from video using smart strategy

        Args:
            video_path: Path to input video
            output_dir: Directory to save sampled frames (optional)

        Returns:
            Tuple of (frames, frame_indices, metadata)
        """
        logger.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video metadata:")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - FPS: {fps:.2f}")
        logger.info(f"  - Resolution: {width}x{height}")
        logger.info(f"  - Duration: {duration:.2f} seconds")

        # Calculate sampling strategy
        sample_interval, num_samples = self._calculate_sampling_strategy(
            total_frames, fps, duration
        )

        logger.info(f"Sampling strategy:")
        logger.info(f"  - Interval: every {sample_interval} frames ({sample_interval/fps:.2f} seconds)")
        logger.info(f"  - Expected samples: {num_samples}")

        # Sample frames
        sampled_frames = []
        frame_indices = []

        frame_idx = 0
        with tqdm(total=num_samples, desc="Sampling frames") as pbar:
            while len(sampled_frames) < num_samples:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                sampled_frames.append(frame.copy())
                frame_indices.append(frame_idx)

                frame_idx += sample_interval
                pbar.update(1)

        cap.release()

        logger.info(f"Successfully sampled {len(sampled_frames)} frames")

        # Save debug visualization if requested
        if self.save_debug and output_dir:
            self._save_debug_visualization(sampled_frames, frame_indices,
                                          output_dir, fps)

        # Prepare metadata
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'resolution': (width, height),
            'sample_interval': sample_interval,
            'num_samples': len(sampled_frames),
            'frame_indices': frame_indices
        }

        return sampled_frames, frame_indices, metadata

    def _calculate_sampling_strategy(self,
                                     total_frames: int,
                                     fps: float,
                                     duration: float) -> Tuple[int, int]:
        """
        Calculate optimal sampling interval and expected number of samples

        Strategy:
        - Short videos (<60s): Sample densely to reach target_samples
        - Long videos (>=60s): Sample every min_interval_seconds
        - Always ensure at least target_samples or all frames (whichever is smaller)

        Args:
            total_frames: Total number of frames in video
            fps: Frames per second
            duration: Video duration in seconds

        Returns:
            Tuple of (sample_interval, num_samples)
        """
        if duration <= 60:
            # Short video: sample densely
            sample_interval = max(1, total_frames // self.target_samples)
            num_samples = min(self.target_samples, total_frames // sample_interval)
        else:
            # Long video: sample at fixed time intervals
            sample_interval = max(1, int(fps * self.min_interval_seconds))
            num_samples = min(self.target_samples, total_frames // sample_interval)

        # Ensure we have enough samples
        if num_samples < self.target_samples and sample_interval > 1:
            sample_interval = max(1, total_frames // self.target_samples)
            num_samples = min(self.target_samples, total_frames // sample_interval)

        return sample_interval, num_samples

    def _save_debug_visualization(self,
                                  frames: List[np.ndarray],
                                  frame_indices: List[int],
                                  output_dir: str,
                                  fps: float):
        """
        Save debug visualization of sampled frames

        Creates a grid image showing all sampled frames as thumbnails

        Args:
            frames: List of sampled frames
            frame_indices: List of frame indices
            output_dir: Output directory
            fps: Video FPS
        """
        output_path = Path(output_dir) / "sampled_frames"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving sampled frames to {output_path}")

        # Save individual frames
        for idx, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            timestamp = frame_idx / fps
            filename = f"frame_{frame_idx:06d}_t{timestamp:.2f}s.jpg"
            cv2.imwrite(str(output_path / filename), frame)

        # Create grid visualization
        grid_image = self._create_grid_visualization(frames, frame_indices, fps)
        grid_path = Path(output_dir) / "analysis" / "sampled_frames_grid.jpg"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(grid_path), grid_image)

        logger.info(f"Saved grid visualization to {grid_path}")

    def _create_grid_visualization(self,
                                   frames: List[np.ndarray],
                                   frame_indices: List[int],
                                   fps: float,
                                   thumbnail_size: Tuple[int, int] = (160, 90)) -> np.ndarray:
        """
        Create a grid visualization of sampled frames

        Args:
            frames: List of frames
            frame_indices: List of frame indices
            fps: Video FPS
            thumbnail_size: Size of each thumbnail (width, height)

        Returns:
            Grid image as numpy array
        """
        num_frames = len(frames)

        # Calculate grid dimensions
        cols = min(10, num_frames)
        rows = (num_frames + cols - 1) // cols

        # Create blank canvas
        thumb_w, thumb_h = thumbnail_size
        canvas_w = cols * thumb_w
        canvas_h = rows * thumb_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Place thumbnails
        for idx, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            row = idx // cols
            col = idx % cols

            # Resize frame to thumbnail
            thumbnail = cv2.resize(frame, thumbnail_size)

            # Add timestamp text
            timestamp = frame_idx / fps
            text = f"{timestamp:.1f}s"
            cv2.putText(thumbnail, text, (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Place on canvas
            y_start = row * thumb_h
            x_start = col * thumb_w
            canvas[y_start:y_start+thumb_h, x_start:x_start+thumb_w] = thumbnail

        return canvas


def test_frame_sampler():
    """Test function for frame sampler"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_sampler.py <video_path>")
        return

    video_path = sys.argv[1]
    output_dir = "/tmp/video-changer/output"

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create sampler
    sampler = FrameSampler(target_samples=200,
                          min_interval_seconds=1.5,
                          save_debug=True)

    # Sample frames
    frames, indices, metadata = sampler.sample_frames(video_path, output_dir)

    print("\n" + "="*60)
    print("SAMPLING COMPLETE")
    print("="*60)
    print(f"Total sampled frames: {len(frames)}")
    print(f"Frame indices: {indices[:10]}... (showing first 10)")
    print(f"Video duration: {metadata['duration']:.2f} seconds")
    print(f"Sampling interval: {metadata['sample_interval']} frames")
    print(f"\nCheck output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    test_frame_sampler()
