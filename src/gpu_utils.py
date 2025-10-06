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


def get_gpu_utilization(gpu_id: int) -> Optional[int]:
    """
    Get GPU utilization percentage for a specific GPU

    Args:
        gpu_id: GPU device ID

    Returns:
        GPU utilization percentage (0-100) or None if failed
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            utilization = int(result.stdout.strip())
            return utilization
    except Exception as e:
        logger.warning(f"Failed to get GPU {gpu_id} utilization: {e}")

    return None


def get_gpu_load_score(gpu_id: int) -> float:
    """
    Calculate a load score for a GPU based on memory usage and utilization
    Lower score = better (less loaded)

    Args:
        gpu_id: GPU device ID

    Returns:
        Load score (lower is better), or float('inf') if failed
    """
    mem_info = get_gpu_memory_info(gpu_id)
    utilization = get_gpu_utilization(gpu_id)

    if mem_info is None and utilization is None:
        return float('inf')

    # Calculate memory usage percentage
    if mem_info:
        mem_usage_pct = (mem_info['used_mb'] / mem_info['total_mb']) * 100 if mem_info['total_mb'] > 0 else 100
    else:
        mem_usage_pct = 50  # Default if not available

    # Use actual utilization or default
    gpu_util_pct = utilization if utilization is not None else 50

    # Combined score: 60% weight on utilization, 40% weight on memory
    # Lower score is better
    load_score = (gpu_util_pct * 0.6) + (mem_usage_pct * 0.4)

    return load_score


def select_best_gpus(num_gpus: Optional[int] = None) -> List[int]:
    """
    Select the best GPUs to use based on load (utilization + memory usage)
    Automatically selects the least loaded GPU(s)

    Args:
        num_gpus: Number of GPUs to select (None = use only 1 best GPU)

    Returns:
        List of selected GPU IDs
    """
    all_gpus = get_available_gpus()

    if not all_gpus:
        return []

    # Default to selecting 1 GPU if not specified
    if num_gpus is None:
        num_gpus = 1

    # Get load info for each GPU
    gpu_info = []
    for gpu_id in all_gpus:
        load_score = get_gpu_load_score(gpu_id)
        mem_info = get_gpu_memory_info(gpu_id)
        utilization = get_gpu_utilization(gpu_id)

        gpu_info.append({
            'id': gpu_id,
            'load_score': load_score,
            'free_mb': mem_info['free_mb'] if mem_info else 0,
            'utilization': utilization if utilization is not None else -1
        })

    # Sort by load score (ascending - lower is better)
    gpu_info.sort(key=lambda x: x['load_score'])

    # Select requested number of GPUs
    selected = [g['id'] for g in gpu_info[:num_gpus]]

    logger.info(f"Auto-selected {len(selected)} least-loaded GPU(s): {selected}")
    for gpu_id in selected:
        gpu = next((g for g in gpu_info if g['id'] == gpu_id), None)
        if gpu:
            if gpu['utilization'] >= 0:
                logger.info(f"  GPU {gpu_id}: {gpu['utilization']}% util, {gpu['free_mb']} MB free, load score: {gpu['load_score']:.1f}")
            else:
                logger.info(f"  GPU {gpu_id}: {gpu['free_mb']} MB free")

    return selected
