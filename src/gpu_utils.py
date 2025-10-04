"""
GPU utilities for multi-GPU processing
"""
import logging
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """
    Detect all available CUDA GPUs

    Returns:
        List of GPU device IDs
    """
    try:
        # Try using nvidia-smi to detect GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            logger.info(f"Detected {len(gpu_ids)} GPUs: {gpu_ids}")
            return gpu_ids
        else:
            logger.warning("nvidia-smi failed, trying torch method")
    except Exception as e:
        logger.warning(f"nvidia-smi detection failed: {e}, trying torch method")

    # Fallback to torch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_ids = list(range(gpu_count))
            logger.info(f"Detected {gpu_count} GPUs via torch: {gpu_ids}")
            return gpu_ids
        else:
            logger.warning("No CUDA GPUs detected via torch")
    except Exception as e:
        logger.warning(f"Torch detection failed: {e}")

    # No GPUs found
    logger.warning("No GPUs detected, will use CPU")
    return []


def get_gpu_memory_info(gpu_id: int) -> Optional[dict]:
    """
    Get memory information for a specific GPU

    Args:
        gpu_id: GPU device ID

    Returns:
        Dictionary with memory info or None if failed
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            total, used, free = map(int, result.stdout.strip().split(','))
            return {
                'total_mb': total,
                'used_mb': used,
                'free_mb': free
            }
    except Exception as e:
        logger.warning(f"Failed to get GPU {gpu_id} memory info: {e}")

    return None


def select_best_gpus(num_gpus: Optional[int] = None) -> List[int]:
    """
    Select the best GPUs to use based on available memory

    Args:
        num_gpus: Number of GPUs to select (None = all available)

    Returns:
        List of selected GPU IDs
    """
    all_gpus = get_available_gpus()

    if not all_gpus:
        return []

    # Get memory info for each GPU
    gpu_info = []
    for gpu_id in all_gpus:
        mem_info = get_gpu_memory_info(gpu_id)
        if mem_info:
            gpu_info.append({
                'id': gpu_id,
                'free_mb': mem_info['free_mb']
            })

    if not gpu_info:
        # If we can't get memory info, just return all GPUs
        selected = all_gpus if num_gpus is None else all_gpus[:num_gpus]
        logger.info(f"Selected GPUs (no memory info): {selected}")
        return selected

    # Sort by free memory (descending)
    gpu_info.sort(key=lambda x: x['free_mb'], reverse=True)

    # Select requested number of GPUs
    if num_gpus is None:
        selected = [g['id'] for g in gpu_info]
    else:
        selected = [g['id'] for g in gpu_info[:num_gpus]]

    logger.info(f"Selected {len(selected)} GPUs: {selected}")
    for gpu_id in selected:
        mem = next((g for g in gpu_info if g['id'] == gpu_id), None)
        if mem:
            logger.info(f"  GPU {gpu_id}: {mem['free_mb']} MB free")

    return selected
