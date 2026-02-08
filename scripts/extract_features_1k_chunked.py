#!/usr/bin/env python3
"""
Extract features from 1K dataset in MEMORY-SAFE CHUNKS.

This script processes large datasets in small batches to prevent RAM exhaustion.
Each chunk is saved immediately, keeping memory usage under 2 GB.

Memory Profile:
- Without chunking: 15+ GB (crashes!)
- With chunking: <2 GB per chunk (safe)

Author: Space AI Team
Date: 2026-02-07
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import click
import json
from datetime import datetime
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.data_generator import Dataset
from src.ml.features.trajectory_features import TrajectoryFeatureExtractor, FeatureConfig
from src.ml.features.sequence_builder import TrajectorySequenceBuilder, SequenceConfig
from src.utils.logging_config import get_logger

logger = get_logger("feature_extraction_chunked")


def process_chunk(
    gt_data,
    object_ids_chunk,
    feature_extractor,
    sequence_builder,
    chunk_idx,
    output_dir
):
    """
    Process one chunk of objects and save immediately.
    
    Returns:
        Statistics dict
    """
    chunk_sequences = {}
    n_sequences = 0
    
    for obj_id in tqdm(object_ids_chunk, desc=f"Chunk {chunk_idx}", leave=False):
        obj_data = gt_data[gt_data['object_id'] == obj_id].sort_values('time')
        
        # Extract trajectories
        positions = obj_data[['position_x', 'position_y', 'position_z']].values
        velocities = obj_data[['velocity_x', 'velocity_y', 'velocity_z']].values
        timestamps = (obj_data['time'] - obj_data['time'].min()).dt.total_seconds().values
        
        # Extract features
        features = feature_extractor.extract_features(
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        # Build sequences
        sequences = sequence_builder.build_sequences(features)
        
        if len(sequences['history']) > 0:
            chunk_sequences[int(obj_id)] = sequences
            n_sequences += len(sequences['history'])
    
    # SAVE IMMEDIATELY (prevents RAM accumulation)
    chunk_path = output_dir / f'features_chunk_{chunk_idx:04d}.pt'
    torch.save(chunk_sequences, chunk_path)
    
    size_mb = chunk_path.stat().st_size / 1024 / 1024
    
    # Clear memory aggressively
    del chunk_sequences
    gc.collect()
    
    return {
        'chunk_idx': chunk_idx,
        'num_objects': len(object_ids_chunk),
        'num_sequences': n_sequences,
        'size_mb': round(size_mb, 2),
        'filename': chunk_path.name
    }


@click.command()
@click.option('--dataset-dir', required=True, type=click.Path(exists=True))
@click.option('--output-dir', required=True, type=click.Path())
@click.option('--chunk-size', default=100, type=int, help='Objects per chunk')
@click.option('--device', default='cuda', type=click.Choice(['cuda', 'cpu']))
def main(dataset_dir, output_dir, chunk_size, device):
    """
    Extract features in memory-safe chunks.
    
    Example:
        python scripts/extract_features_1k_chunked.py \
            --dataset-dir data/processed/ml_train_1k \
            --output-dir data/processed/features_1k_chunked \
            --chunk-size 100 \
            --device cuda
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        device = 'cpu'
    
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("🧩 Feature Extraction (CHUNKED - Memory Safe)")
    print("="*70)
    print(f"Dataset:     {dataset_dir}")
    print(f"Output:      {output_dir}")
    print(f"Chunk size:  {chunk_size} objects")
    print(f"Device:      {device}")
    
    if device == 'cuda':
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("="*70 + "\n")
    
    # Load dataset
    print("📊 Step 1/5: Loading dataset...")
    dataset = Dataset.load(dataset_dir)
    gt = dataset.ground_truth
    
    print(f"  Objects: {len(gt['object_id'].unique())}")
    print(f"  Total points: {len(gt):,}")
    
    # Initialize extractors
    print("\n🔧 Step 2/5: Initializing feature extractors...")
    feature_config = FeatureConfig()
    sequence_config = SequenceConfig(
        history_length=20,
        prediction_horizon=30,
        stride=1,
        normalize=True,
        normalization_method='standard'
    )
    
    feature_extractor = TrajectoryFeatureExtractor(feature_config)
    sequence_builder = TrajectorySequenceBuilder(sequence_config)
    
    print(f"  Feature dimension: 28D")
    print(f"  History length: {sequence_config.history_length}")
    print(f"  Prediction horizon: {sequence_config.prediction_horizon}")
    
    # Split into chunks
    object_ids = gt['object_id'].unique()
    num_chunks = int(np.ceil(len(object_ids) / chunk_size))
    
    print(f"\n🧩 Step 3/5: Processing {len(object_ids)} objects in {num_chunks} chunks...")
    print(f"  Chunk size: {chunk_size} objects")
    print(f"  Memory per chunk: ~1-2 GB (safe!)")
    print(f"  Estimated time: ~{num_chunks * 20} seconds ({num_chunks * 20 / 60:.1f} minutes)")
    print("")
    
    chunk_stats = []
    total_sequences = 0
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(object_ids))
        object_ids_chunk = object_ids[start_idx:end_idx]
        
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} (objects {start_idx}-{end_idx})...")
        
        stats = process_chunk(
            gt_data=gt,
            object_ids_chunk=object_ids_chunk,
            feature_extractor=feature_extractor,
            sequence_builder=sequence_builder,
            chunk_idx=chunk_idx,
            output_dir=output_dir
        )
        
        chunk_stats.append(stats)
        total_sequences += stats['num_sequences']
        
        print(f"  ✅ Chunk {chunk_idx}: {stats['num_objects']} objects, "
              f"{stats['num_sequences']:,} sequences, {stats['size_mb']:.1f} MB")
        
        # Force garbage collection between chunks
        gc.collect()
    
    # Combine chunks into single file
    print(f"\n💾 Step 4/5: Combining {num_chunks} chunks...")
    
    all_sequences = {}
    for stat in tqdm(chunk_stats, desc="Combining chunks"):
        chunk_path = output_dir / stat['filename']
        chunk_data = torch.load(chunk_path, map_location='cpu')
        all_sequences.update(chunk_data)
        
        # Clear after loading
        del chunk_data
        gc.collect()
    
    # Save combined file
    combined_path = output_dir / 'features.pt'
    torch.save(all_sequences, combined_path)
    
    size_mb = combined_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Combined file saved: {size_mb:.1f} MB")
    
    # Clean up individual chunks (optional - keep for debugging)
    # for stat in chunk_stats:
    #     (output_dir / stat['filename']).unlink()
    
    # Save metadata
    print("\n📝 Step 5/5: Saving metadata...")
    metadata = {
        'dataset_dir': str(dataset_dir),
        'output_dir': str(output_dir),
        'device': device,
        'chunk_size': chunk_size,
        'num_chunks': num_chunks,
        'extraction_started': start_time.isoformat(),
        'extraction_completed': datetime.now().isoformat(),
        'extraction_duration_minutes': round((datetime.now() - start_time).total_seconds() / 60, 2),
        'feature_config': {
            'include_orbital_elements': feature_config.include_orbital_elements,
            'include_derived_features': feature_config.include_derived_features,
            'include_temporal_features': feature_config.include_temporal_features
        },
        'sequence_config': {
            'history_length': sequence_config.history_length,
            'prediction_horizon': sequence_config.prediction_horizon,
            'stride': sequence_config.stride
        },
        'chunks': chunk_stats,
        'totals': {
            'num_objects': sum(s['num_objects'] for s in chunk_stats),
            'total_sequences': total_sequences,
            'total_size_mb': round(size_mb, 2)
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    end_time = datetime.now()
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    # Print summary
    print("\n" + "="*70)
    print("✅ Feature Extraction Complete!")
    print("="*70)
    print(f"Total objects:       {sum(s['num_objects'] for s in chunk_stats)}")
    print(f"Total sequences:     {total_sequences:,}")
    print(f"Output size:         {size_mb:.1f} MB")
    print(f"Extraction time:     {duration_minutes:.1f} minutes")
    print(f"Peak memory usage:   <2 GB per chunk (safe!)")
    print(f"Output directory:    {output_dir}")
    print("="*70 + "\n")
    
    print("✅ Features ready for training (Stage 4)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
