"""
Dominant Face Analyzer

Analyzes sampled frames to identify the most dominant face in the video
based on frequency, size, and clarity metrics.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import pickle

from face_detector import DetectedFace, FaceDetector

logger = logging.getLogger(__name__)


@dataclass
class FaceIdentity:
    """Represents a unique face identity tracked across frames"""
    face_id: int
    appearances: List[DetectedFace] = field(default_factory=list)
    avg_embedding: Optional[np.ndarray] = None
    frequency_score: float = 0.0
    size_score: float = 0.0
    clarity_score: float = 0.0
    total_score: float = 0.0

    def add_appearance(self, face: DetectedFace):
        """Add a new appearance of this face"""
        face.face_id = self.face_id
        self.appearances.append(face)

    def compute_average_embedding(self):
        """Compute average embedding from all appearances"""
        embeddings = [f.embedding for f in self.appearances
                     if f.embedding is not None]
        if embeddings:
            self.avg_embedding = np.mean(embeddings, axis=0)
            # Normalize
            self.avg_embedding = self.avg_embedding / np.linalg.norm(self.avg_embedding)

    def get_best_face(self) -> DetectedFace:
        """Get the best quality appearance of this face"""
        # Sort by confidence and size
        sorted_faces = sorted(self.appearances,
                            key=lambda f: f.confidence * (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                            reverse=True)
        return sorted_faces[0] if sorted_faces else None


class DominantFaceAnalyzer:
    """Analyzes faces across frames to find the dominant one"""

    def __init__(self,
                 weight_frequency: float = 0.4,
                 weight_size: float = 0.4,
                 weight_clarity: float = 0.2,
                 similarity_threshold: float = 0.4):
        """
        Initialize dominant face analyzer

        Args:
            weight_frequency: Weight for appearance frequency
            weight_size: Weight for face size
            weight_clarity: Weight for face clarity/quality
            similarity_threshold: Threshold for face matching (cosine similarity)
        """
        self.weight_frequency = weight_frequency
        self.weight_size = weight_size
        self.weight_clarity = weight_clarity
        self.similarity_threshold = similarity_threshold

        self.face_identities: Dict[int, FaceIdentity] = {}
        self.next_face_id = 0

    def analyze(self,
                detection_results: Dict[int, List[DetectedFace]],
                frames: List[np.ndarray] = None,
                save_debug: bool = False,
                output_dir: str = None) -> Tuple[FaceIdentity, List[FaceIdentity]]:
        """
        Analyze detected faces to find the dominant one

        Args:
            detection_results: Dictionary mapping frame_idx -> list of DetectedFace
            frames: Optional list of frames for debug visualization
            save_debug: Whether to save debug visualizations
            output_dir: Output directory

        Returns:
            Tuple of (dominant_face_identity, all_face_identities)
        """
        logger.info("Analyzing faces to find dominant identity...")

        # Step 1: Track faces across frames
        self._track_faces(detection_results)

        # Step 2: Compute average embeddings for each identity
        for identity in self.face_identities.values():
            identity.compute_average_embedding()

        # Step 3: Compute scores for each identity
        self._compute_scores(detection_results)

        # Step 4: Find dominant face
        dominant_identity = self._find_dominant_face()

        # Log results
        logger.info(f"\nFound {len(self.face_identities)} unique face identities")
        logger.info(f"\nTop 5 faces by score:")
        sorted_identities = sorted(self.face_identities.values(),
                                  key=lambda x: x.total_score,
                                  reverse=True)
        for idx, identity in enumerate(sorted_identities[:5]):
            logger.info(f"  Face {identity.face_id}:")
            logger.info(f"    - Appearances: {len(identity.appearances)}")
            logger.info(f"    - Frequency score: {identity.frequency_score:.3f}")
            logger.info(f"    - Size score: {identity.size_score:.3f}")
            logger.info(f"    - Clarity score: {identity.clarity_score:.3f}")
            logger.info(f"    - Total score: {identity.total_score:.3f}")

        logger.info(f"\n✓ Dominant face: Face ID {dominant_identity.face_id}")
        logger.info(f"  - Total appearances: {len(dominant_identity.appearances)}")
        logger.info(f"  - Total score: {dominant_identity.total_score:.3f}")

        # Save debug visualizations
        if save_debug and output_dir and frames is not None:
            self._save_debug_visualizations(detection_results, frames,
                                           dominant_identity, output_dir)

        return dominant_identity, sorted_identities

    def _track_faces(self, detection_results: Dict[int, List[DetectedFace]]):
        """
        Track faces across frames to create face identities

        Uses embedding similarity to match faces across frames
        """
        logger.info("Tracking faces across frames...")

        from tqdm import tqdm

        for frame_idx in tqdm(sorted(detection_results.keys()), desc="Tracking faces"):
            faces = detection_results[frame_idx]

            for face in faces:
                # Try to match with existing identities
                matched_identity = self._match_face(face)

                if matched_identity is not None:
                    # Add to existing identity
                    matched_identity.add_appearance(face)
                else:
                    # Create new identity
                    new_identity = FaceIdentity(face_id=self.next_face_id)
                    new_identity.add_appearance(face)
                    self.face_identities[self.next_face_id] = new_identity
                    self.next_face_id += 1

        logger.info(f"Tracked {len(self.face_identities)} unique identities")

    def _match_face(self, face: DetectedFace) -> Optional[FaceIdentity]:
        """
        Try to match a face to an existing identity

        Args:
            face: Detected face to match

        Returns:
            Matched FaceIdentity or None if no match found
        """
        if face.embedding is None:
            return None

        best_match = None
        best_similarity = self.similarity_threshold

        face_emb = face.embedding / np.linalg.norm(face.embedding)

        for identity in self.face_identities.values():
            if identity.avg_embedding is None:
                # Use embedding from most recent appearance
                if identity.appearances and identity.appearances[-1].embedding is not None:
                    ref_emb = identity.appearances[-1].embedding
                    ref_emb = ref_emb / np.linalg.norm(ref_emb)
                else:
                    continue
            else:
                ref_emb = identity.avg_embedding

            # Compute similarity
            similarity = np.dot(face_emb, ref_emb)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = identity

        return best_match

    def _compute_scores(self, detection_results: Dict[int, List[DetectedFace]]):
        """
        Compute frequency, size, and clarity scores for each identity
        """
        logger.info("Computing scores for each identity...")

        total_frames = len(detection_results)
        all_face_sizes = []
        all_confidences = []

        # Collect all face sizes and confidences for normalization
        for faces in detection_results.values():
            for face in faces:
                width = face.bbox[2] - face.bbox[0]
                height = face.bbox[3] - face.bbox[1]
                size = width * height
                all_face_sizes.append(size)
                all_confidences.append(face.confidence)

        max_size = max(all_face_sizes) if all_face_sizes else 1
        max_conf = max(all_confidences) if all_confidences else 1

        # Compute scores for each identity
        for identity in self.face_identities.values():
            # Frequency score: proportion of frames where face appears
            identity.frequency_score = len(identity.appearances) / total_frames

            # Size score: average normalized size
            sizes = []
            for face in identity.appearances:
                width = face.bbox[2] - face.bbox[0]
                height = face.bbox[3] - face.bbox[1]
                size = width * height
                sizes.append(size / max_size)
            identity.size_score = np.mean(sizes) if sizes else 0

            # Clarity score: average confidence
            confidences = [f.confidence for f in identity.appearances]
            identity.clarity_score = np.mean(confidences) / max_conf if confidences else 0

            # Total weighted score
            identity.total_score = (
                self.weight_frequency * identity.frequency_score +
                self.weight_size * identity.size_score +
                self.weight_clarity * identity.clarity_score
            )

    def _find_dominant_face(self) -> FaceIdentity:
        """Find the face identity with highest total score"""
        dominant = max(self.face_identities.values(),
                      key=lambda x: x.total_score)
        return dominant

    def _save_debug_visualizations(self,
                                   detection_results: Dict[int, List[DetectedFace]],
                                   frames: List[np.ndarray],
                                   dominant_identity: FaceIdentity,
                                   output_dir: str):
        """Save debug visualizations"""
        output_path = Path(output_dir) / "analysis"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dominant face crops
        self._save_dominant_face_crops(dominant_identity, frames,
                                       detection_results, output_dir)

        # Save face tracking visualization
        self._save_tracking_visualization(detection_results, frames,
                                        dominant_identity, output_dir)

        # Save statistics report
        self._save_statistics_report(output_dir)

    def _save_dominant_face_crops(self,
                                 dominant_identity: FaceIdentity,
                                 frames: List[np.ndarray],
                                 detection_results: Dict[int, List[DetectedFace]],
                                 output_dir: str):
        """Save crops of the dominant face from different frames"""
        output_path = Path(output_dir) / "analysis" / "dominant_face_crops"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving dominant face crops to {output_path}")

        # Get frame index mapping
        frame_indices = sorted(detection_results.keys())
        frame_idx_to_array_idx = {idx: i for i, idx in enumerate(frame_indices)}

        # Sample diverse appearances (different angles, lighting, etc.)
        appearances = dominant_identity.appearances
        num_samples = min(20, len(appearances))
        sample_indices = np.linspace(0, len(appearances)-1, num_samples, dtype=int)

        from face_detector import FaceDetector
        detector = FaceDetector()

        for idx, sample_idx in enumerate(sample_indices):
            face = appearances[sample_idx]
            frame_idx = face.frame_idx

            if frame_idx not in frame_idx_to_array_idx:
                continue

            array_idx = frame_idx_to_array_idx[frame_idx]
            frame = frames[array_idx]

            # Extract face crop
            face_crop = detector.extract_face_crop(frame, face, padding=0.3)

            # Save
            filename = f"dominant_face_{idx:03d}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_path / filename), face_crop)

        # Also save the best quality crop
        best_face = dominant_identity.get_best_face()
        if best_face and best_face.frame_idx in frame_idx_to_array_idx:
            array_idx = frame_idx_to_array_idx[best_face.frame_idx]
            frame = frames[array_idx]
            best_crop = detector.extract_face_crop(frame, best_face, padding=0.3)
            cv2.imwrite(str(output_path / "dominant_face_best.jpg"), best_crop)

        logger.info(f"Saved {num_samples+1} dominant face crops")

    def _save_tracking_visualization(self,
                                    detection_results: Dict[int, List[DetectedFace]],
                                    frames: List[np.ndarray],
                                    dominant_identity: FaceIdentity,
                                    output_dir: str):
        """Save visualization showing tracked faces"""
        output_path = Path(output_dir) / "analysis"
        output_path.mkdir(parents=True, exist_ok=True)

        # Get frame index mapping
        frame_indices = sorted(detection_results.keys())
        frame_idx_to_array_idx = {idx: i for i, idx in enumerate(frame_indices)}

        # Create color map for face identities
        num_identities = len(self.face_identities)
        colors = self._generate_colors(num_identities)
        id_to_color = {identity.face_id: colors[i]
                      for i, identity in enumerate(self.face_identities.values())}

        # Sample frames to visualize
        num_vis_frames = min(16, len(frames))
        vis_frame_indices = np.linspace(0, len(frames)-1, num_vis_frames, dtype=int)

        vis_frames = []
        for array_idx in vis_frame_indices:
            frame_idx = frame_indices[array_idx]
            frame = frames[array_idx].copy()

            # Draw all faces with their identity colors
            for face in detection_results[frame_idx]:
                bbox = face.bbox
                color = id_to_color.get(face.face_id, (128, 128, 128))

                # Thicker border for dominant face
                thickness = 4 if face.face_id == dominant_identity.face_id else 2

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            color, thickness)

                # Label
                label = f"ID{face.face_id}"
                if face.face_id == dominant_identity.face_id:
                    label += " (DOMINANT)"

                cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Resize for grid
            vis_frames.append(cv2.resize(frame, (320, 180)))

        # Create grid
        grid = self._create_grid(vis_frames, cols=4)
        cv2.imwrite(str(output_path / "face_tracking_visualization.jpg"), grid)

        logger.info("Saved face tracking visualization")

    def _save_statistics_report(self, output_dir: str):
        """Save text report with statistics"""
        output_path = Path(output_dir) / "analysis" / "dominant_face_report.txt"

        sorted_identities = sorted(self.face_identities.values(),
                                  key=lambda x: x.total_score,
                                  reverse=True)

        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DOMINANT FACE ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total unique faces found: {len(self.face_identities)}\n\n")

            f.write("Scoring weights:\n")
            f.write(f"  - Frequency: {self.weight_frequency}\n")
            f.write(f"  - Size: {self.weight_size}\n")
            f.write(f"  - Clarity: {self.weight_clarity}\n\n")

            f.write("="*60 + "\n")
            f.write("ALL FACE IDENTITIES (sorted by score):\n")
            f.write("="*60 + "\n\n")

            for rank, identity in enumerate(sorted_identities, 1):
                f.write(f"Rank {rank}: Face ID {identity.face_id}\n")
                f.write(f"  - Appearances: {len(identity.appearances)}\n")
                f.write(f"  - Frequency score: {identity.frequency_score:.4f}\n")
                f.write(f"  - Size score: {identity.size_score:.4f}\n")
                f.write(f"  - Clarity score: {identity.clarity_score:.4f}\n")
                f.write(f"  - TOTAL SCORE: {identity.total_score:.4f}\n")
                if rank == 1:
                    f.write(f"  >>> DOMINANT FACE <<<\n")
                f.write("\n")

        logger.info(f"Saved statistics report to {output_path}")

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def _create_grid(self, images: List[np.ndarray], cols: int = 4) -> np.ndarray:
        """Create grid layout from images"""
        rows = (len(images) + cols - 1) // cols
        h, w = images[0].shape[:2]

        # Pad with blank images if needed
        while len(images) < rows * cols:
            images.append(np.zeros_like(images[0]))

        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = images[i*cols:(i+1)*cols]
            grid_rows.append(np.hstack(row_images))

        grid = np.vstack(grid_rows)
        return grid


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use main.py to run the full pipeline.")
