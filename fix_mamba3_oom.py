#!/usr/bin/env python3
"""
Quick implementation script for Mamba3 OOM fixes on Kaggle.
Run this to set up environment and verify configs.
"""

import os
import sys
from pathlib import Path

def check_configs():
    """Check and report current training config setup."""
    config_dir = Path("src/f5_tts/configs")
    
    print("=" * 70)
    print("MAMBA3 KAGGLE OOM FIX - CONFIGURATION CHECK")
    print("=" * 70)
    
    # Look for Mamba3 configs
    mamba3_configs = list(config_dir.glob("*mamba3*"))
    print(f"\n✓ Found {len(mamba3_configs)} Mamba3-related config files:")
    for cfg in mamba3_configs:
        print(f"  - {cfg.name}")
    
    print("\n" + "=" * 70)
    print("APPLY THESE FIXES:")
    print("=" * 70)
    
    print("\n1. ✅ [ALREADY DONE] Trainer.py: find_unused_parameters=False")
    print("   Location: src/f5_tts/model/trainer.py line 56")
    
    print("\n2. 📝 [YOU NEED TO DO] Model Config: Add checkpoint_activations")
    print("   File: Your Mamba3 config YAML")
    print("   Add under 'model' section:")
    print("   ----")
    print("   model:")
    print("     checkpoint_activations: true")
    print("   ----")
    
    print("\n3. 📝 [YOU NEED TO DO] Accelerate Config: Add mixed_precision")
    print("   File: accelerate_kaggle.yaml")
    print("   Add line:")
    print("   ----")
    print("   mixed_precision: fp16")
    print("   ----")
    
    print("\n4. 📝 [YOU NEED TO DO] Dataset Config: Reduce batch size")
    print("   File: Your Mamba3 config YAML")
    print("   Change under 'datasets' section:")
    print("   ----")
    print("   datasets:")
    print("     batch_size_per_gpu: 16     # REDUCE from 32")
    print("     batch_size_type: 'sample'")
    print("   ----")
    print("   Optional: Add grad accumulation in 'optim' section:")
    print("   ----")
    print("   optim:")
    print("     grad_accumulation_steps: 2")
    print("   ----")
    
    print("\n5. 🔧 [SET BEFORE TRAINING] Environment Variable")
    print("   Set in your training notebook/script:")
    print("   ----")
    print("   import os")
    print("   os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'")
    print("   ----")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. Open your Mamba3 config YAML file")
    print("2. Add checkpoint_activations: true to model section")
    print("3. Open accelerate_kaggle.yaml and add mixed_precision: fp16")
    print("4. Reduce batch_size_per_gpu from 32 to 16")
    print("5. Before training, set PYTORCH_ALLOC_CONF in your notebook")
    print("\nExample training command:")
    print("  python3 -m accelerate.commands.launch \\")
    print("    --config_file accelerate_kaggle.yaml \\")
    print("    src/f5_tts/train/train.py \\")
    print("    --config-name your_mamba3_config")
    print("\n" + "=" * 70)


def setup_env():
    """Set up environment for training."""
    print("\n🔧 Setting PYTORCH_ALLOC_CONF...")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    print(f"   ✓ Set to: {os.environ['PYTORCH_ALLOC_CONF']}")
    print("   This reduces GPU memory fragmentation")
    
    # Check CUDA
    try:
        import torch
        print(f"\n✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  - Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - Shared mem: {getattr(props, 'shared_memory_per_block', 'N/A')} bytes")
        else:
            print("⚠ CUDA not available!")
    except ImportError:
        print("⚠ PyTorch not installed!")


if __name__ == "__main__":
    check_configs()
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_env()

