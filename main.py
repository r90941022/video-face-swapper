#!/usr/bin/env python3
"""
Video Changer - High Quality Face Replacement

Main execution script with step-by-step interactive mode
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import cv2
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from frame_sampler import FrameSampler
from face_detector import FaceDetector
from dominant_analyzer import DominantFaceAnalyzer
from face_swapper import FaceSwapper
from video_processor import VideoProcessor


class VideoChanger:
    """Main application controller"""

    def __init__(self, config_path: str = None):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.setup_logging()

        self.logger = logging.getLogger(__name__)
        self.logger.info("Video Changer initialized")

        # Components (lazy initialization)
        self.frame_sampler = None
        self.face_detector = None
        self.dominant_analyzer = None
        self.face_swapper = None
        self.video_processor = None

        # Data
        self.sampled_frames = None
        self.frame_indices = None
        self.video_metadata = None
        self.detection_results = None
        self.dominant_identity = None
        self.all_identities = None
        self.target_faces = None

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.INFO if self.config['debug']['verbose'] else logging.WARNING

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/video_changer.log')
            ]
        )

    def run(self,
            input_video: str,
            target_images_dir: str,
            output_video: str,
            step_by_step: bool = None):
        """
        Run the complete pipeline

        Args:
            input_video: Path to input video
            target_images_dir: Directory containing target face images
            output_video: Path to output video
            step_by_step: Enable step-by-step mode (None = use config)
        """
        if step_by_step is None:
            step_by_step = self.config['debug']['step_by_step_mode']

        self.logger.info("="*60)
        self.logger.info("VIDEO CHANGER - Starting Pipeline")
        self.logger.info("="*60)
        self.logger.info(f"Input video: {input_video}")
        self.logger.info(f"Target images: {target_images_dir}")
        self.logger.info(f"Output video: {output_video}")
        self.logger.info(f"Mode: {'Step-by-step' if step_by_step else 'Automatic'}")
        self.logger.info("="*60)

        try:
            # Step 1: Frame Sampling
            self.step_1_sample_frames(input_video, step_by_step)

            # Step 2: Face Detection
            self.step_2_detect_faces(step_by_step)

            # Step 3: Dominant Face Analysis
            self.step_3_analyze_dominant_face(step_by_step)

            # Step 4: Prepare Target Faces
            self.step_4_prepare_target_faces(target_images_dir, step_by_step)

            # Step 5: Preview Swap
            self.step_5_preview_swap(step_by_step)

            # Step 6: Process Full Video
            self.step_6_process_video(input_video, output_video, step_by_step)

            self.logger.info("="*60)
            self.logger.info("✓ PIPELINE COMPLETE!")
            self.logger.info("="*60)

        except KeyboardInterrupt:
            self.logger.info("\n\nPipeline interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
            sys.exit(1)

    def step_1_sample_frames(self, input_video: str, step_by_step: bool):
        """Step 1: Sample frames from video"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 1: SMART FRAME SAMPLING")
        self.logger.info("="*60)

        # Initialize sampler
        self.frame_sampler = FrameSampler(
            target_samples=self.config['frame_sampling']['target_samples'],
            min_interval_seconds=self.config['frame_sampling']['min_interval_seconds'],
            save_debug=self.config['frame_sampling']['save_debug_images']
        )

        # Sample frames
        output_dir = "output"
        self.sampled_frames, self.frame_indices, self.video_metadata = \
            self.frame_sampler.sample_frames(input_video, output_dir)

        self.logger.info(f"\n✓ Sampled {len(self.sampled_frames)} frames")
        self.logger.info(f"✓ Check visualization: {output_dir}/analysis/sampled_frames_grid.jpg")

        if step_by_step:
            self._wait_for_confirmation("Continue to face detection?")

    def step_2_detect_faces(self, step_by_step: bool):
        """Step 2: Detect faces in sampled frames"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 2: FACE DETECTION")
        self.logger.info("="*60)

        # Auto-select GPU if using CUDA (V2 Phase 1 feature)
        gpu_id = self.config['hardware']['gpu_id']
        if self.config['hardware']['execution_provider'] == 'cuda':
            # Check if auto-selection is requested
            if gpu_id == 'auto':
                try:
                    from gpu_utils import select_best_gpus
                    import os
                    selected_gpus = select_best_gpus(num_gpus=1)
                    if selected_gpus:
                        gpu_id = selected_gpus[0]
                        # Set CUDA_VISIBLE_DEVICES to selected GPU
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                        self.logger.info(f"Auto-selected GPU: {gpu_id}")
                    else:
                        gpu_id = 0  # Fallback to GPU 0
                        self.logger.warning(f"GPU auto-selection returned no GPUs, using default GPU {gpu_id}")
                except Exception as e:
                    gpu_id = 0  # Fallback to GPU 0
                    self.logger.warning(f"GPU auto-selection failed: {e}, using default GPU {gpu_id}")
            else:
                # Use specified GPU ID
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                self.logger.info(f"Using specified GPU: {gpu_id}")

        # Initialize detector
        self.face_detector = FaceDetector(
            detection_threshold=self.config['dominant_face']['detection_threshold'],
            min_face_size=self.config['dominant_face']['min_face_size'],
            provider=self.config['hardware']['execution_provider']
        )

        # Detect faces
        self.detection_results = self.face_detector.detect_faces_batch(
            self.sampled_frames,
            self.frame_indices,
            save_debug=self.config['debug']['save_intermediate'],
            output_dir="output"
        )

        total_faces = sum(len(faces) for faces in self.detection_results.values())
        self.logger.info(f"\n✓ Detected {total_faces} total face instances")
        self.logger.info(f"✓ Check visualizations: output/detected_faces/")

        if step_by_step:
            self._wait_for_confirmation("Continue to dominant face analysis?")

    def step_3_analyze_dominant_face(self, step_by_step: bool):
        """Step 3: Analyze and identify dominant face"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 3: DOMINANT FACE ANALYSIS")
        self.logger.info("="*60)

        # Initialize analyzer
        self.dominant_analyzer = DominantFaceAnalyzer(
            weight_frequency=self.config['dominant_face']['weight_frequency'],
            weight_size=self.config['dominant_face']['weight_size'],
            weight_clarity=self.config['dominant_face']['weight_clarity']
        )

        # Analyze (with V2 Phase 1: gender filtering support)
        self.dominant_identity, self.all_identities = \
            self.dominant_analyzer.analyze(
                self.detection_results,
                self.sampled_frames,
                save_debug=self.config['debug']['save_intermediate'],
                output_dir="output",
                gender_filter=self.config['dominant_face'].get('gender_filter', False),
                target_gender=self.config['dominant_face'].get('target_gender', None)
            )

        self.logger.info(f"\n✓ Identified dominant face: ID {self.dominant_identity.face_id}")
        self.logger.info(f"✓ Check results:")
        self.logger.info(f"  - Face crops: output/analysis/dominant_face_crops/")
        self.logger.info(f"  - Tracking visualization: output/analysis/face_tracking_visualization.jpg")
        self.logger.info(f"  - Statistics report: output/analysis/dominant_face_report.txt")

        if step_by_step:
            self._wait_for_confirmation("Is this the correct dominant face?")

    def step_4_prepare_target_faces(self, target_images_dir: str, step_by_step: bool):
        """Step 4: Prepare target faces"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 4: PREPARE TARGET FACES")
        self.logger.info("="*60)

        # Initialize swapper
        self.face_swapper = FaceSwapper(
            use_enhancer=self.config['face_processing']['face_enhancer'] is not None,
            enhancer_model=self.config['face_processing']['face_enhancer'],
            blend_ratio=self.config['face_processing']['blend_ratio'],
            face_mask_blur=self.config['face_processing']['face_mask_blur'],
            providers=self._get_providers()
        )

        self.face_swapper.initialize()

        # Find target images
        target_dir = Path(target_images_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        target_image_paths = []

        for ext in image_extensions:
            target_image_paths.extend(target_dir.glob(f'*{ext}'))
            target_image_paths.extend(target_dir.glob(f'*{ext.upper()}'))

        target_image_paths = [str(p) for p in target_image_paths]

        if len(target_image_paths) == 0:
            raise ValueError(f"No images found in {target_images_dir}")

        self.logger.info(f"Found {len(target_image_paths)} target images")

        # Extract target faces
        self.target_faces = self.face_swapper.prepare_target_faces(
            target_image_paths,
            self.face_detector
        )

        if len(self.target_faces) == 0:
            raise ValueError("No faces detected in target images")

        self.logger.info(f"\n✓ Prepared {len(self.target_faces)} target faces")

        if step_by_step:
            self._wait_for_confirmation("Continue to preview?")

    def step_5_preview_swap(self, step_by_step: bool):
        """Step 5: Create preview of face swap"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 5: PREVIEW FACE SWAP")
        self.logger.info("="*60)

        # Initialize video processor
        self.video_processor = VideoProcessor(
            detector=self.face_detector,
            swapper=self.face_swapper,
            keep_fps=self.config['video']['keep_fps'],
            keep_audio=self.config['video']['keep_audio'],
            output_quality=self.config['video']['output_quality'],
            batch_size=self.config['video']['batch_size']
        )

        # Create preview comparison
        preview_path = "output/analysis/preview_comparison.jpg"
        self.video_processor.create_preview_comparison(
            self.sampled_frames,
            self.frame_indices,
            self.dominant_identity,
            self.target_faces,
            preview_path,
            num_preview=10
        )

        self.logger.info(f"\n✓ Preview created: {preview_path}")
        self.logger.info("✓ Check the side-by-side comparison")

        if step_by_step:
            self._wait_for_confirmation("Proceed with full video processing?")

    def step_6_process_video(self, input_video: str, output_video: str, step_by_step: bool):
        """Step 6: Process full video"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 6: PROCESS FULL VIDEO")
        self.logger.info("="*60)
        self.logger.info("This may take a while depending on video length...")

        # Process video
        stats = self.video_processor.process_video(
            input_video,
            output_video,
            self.dominant_identity,
            self.target_faces
        )

        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("="*60)
        self.logger.info(f"Total frames processed: {stats['total_frames']}")
        self.logger.info(f"Frames with face swapped: {stats['swapped_frames']}")
        self.logger.info(f"Frames skipped (no face): {stats['skipped_frames']}")
        self.logger.info(f"Frames with errors: {stats['error_frames']}")
        self.logger.info(f"\n✓ Output video: {output_video}")

    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime providers from config"""
        provider = self.config['hardware']['execution_provider']

        if provider == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif provider == 'coreml':
            return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    def _wait_for_confirmation(self, message: str):
        """Wait for user confirmation in step-by-step mode"""
        print("\n" + "-"*60)
        response = input(f"{message} (y/n/q): ").strip().lower()

        if response == 'q':
            print("Quitting...")
            sys.exit(0)
        elif response == 'n':
            print("Stopping pipeline. You can adjust settings and restart.")
            sys.exit(0)
        elif response != 'y':
            print("Invalid response. Assuming 'yes'.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Video Changer - High Quality Face Replacement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step-by-step mode (default)
  python main.py --source input/video.mp4 --target target_images/ --output output/result.mp4

  # Automatic mode (no confirmations)
  python main.py --source input/video.mp4 --target target_images/ --output output/result.mp4 --auto

  # With custom config
  python main.py --source input/video.mp4 --target target_images/ --output output/result.mp4 --config my_config.yaml
        """
    )

    parser.add_argument('--source', '-s', required=True,
                       help='Path to source video')
    parser.add_argument('--target', '-t', required=True,
                       help='Directory containing target face images (3-5 photos recommended)')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output video')
    parser.add_argument('--config', '-c', default=None,
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--auto', action='store_true',
                       help='Run in automatic mode (no step-by-step confirmations)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.source).exists():
        print(f"Error: Source video not found: {args.source}")
        sys.exit(1)

    if not Path(args.target).exists():
        print(f"Error: Target directory not found: {args.target}")
        sys.exit(1)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Run
    try:
        app = VideoChanger(args.config)

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        app.run(
            input_video=args.source,
            target_images_dir=args.target,
            output_video=args.output,
            step_by_step=not args.auto
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
