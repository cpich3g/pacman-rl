#!/usr/bin/env python3
"""
Test script to verify pacman-rl installation.

Run this after installation to check that all dependencies are correctly set up.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all critical imports work."""
    print("Testing imports...")
    
    tests = []
    
    # Core dependencies
    try:
        import torch
        tests.append(("✓", f"PyTorch {torch.__version__}"))
    except ImportError as e:
        tests.append(("✗", f"PyTorch: {e}"))
    
    try:
        import transformers
        tests.append(("✓", f"Transformers {transformers.__version__}"))
    except ImportError as e:
        tests.append(("✗", f"Transformers: {e}"))
    
    try:
        import trl
        tests.append(("✓", f"TRL {trl.__version__}"))
    except ImportError as e:
        tests.append(("✗", f"TRL: {e}"))
    
    try:
        import unsloth
        tests.append(("✓", "Unsloth"))
    except ImportError as e:
        tests.append(("✗", f"Unsloth: {e}"))
    
    # Web framework dependencies
    try:
        import fastapi
        tests.append(("✓", f"FastAPI {fastapi.__version__}"))
    except ImportError as e:
        tests.append(("✗", f"FastAPI: {e}"))
    
    try:
        import flask
        tests.append(("✓", f"Flask {flask.__version__}"))
    except ImportError as e:
        tests.append(("✗", f"Flask: {e}"))
    
    # OpenEnv dependencies
    try:
        from core.env_server import Environment
        tests.append(("✓", "OpenEnv core.env_server"))
    except ImportError as e:
        tests.append(("✗", f"OpenEnv: {e}"))
    
    try:
        from core.http_env_client import HTTPEnvClient
        tests.append(("✓", "OpenEnv core.http_env_client"))
    except ImportError as e:
        tests.append(("✗", f"OpenEnv HTTP client: {e}"))
    
    # Pac-Man environment
    try:
        from pacman_env.client import PacManEnv
        tests.append(("✓", "Pac-Man Environment"))
    except ImportError as e:
        tests.append(("✗", f"Pac-Man Environment: {e}"))
    
    try:
        from pacman_env.models import PacManAction, PacManObservation
        tests.append(("✓", "Pac-Man Models"))
    except ImportError as e:
        tests.append(("✗", f"Pac-Man Models: {e}"))
    
    # Print results
    print("\nDependency Check:")
    print("-" * 60)
    for status, name in tests:
        print(f"{status} {name}")
    
    # Summary
    passed = sum(1 for s, _ in tests if s == "✓")
    total = len(tests)
    print("-" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All dependencies installed correctly!")
        return True
    else:
        print("\n✗ Some dependencies are missing. Check errors above.")
        return False


def test_docker():
    """Test Docker availability."""
    print("\n" + "=" * 60)
    print("Testing Docker...")
    print("=" * 60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ Docker daemon is running")
                return True
            else:
                print("✗ Docker daemon is not running")
                print("  Start Docker Desktop or run: sudo systemctl start docker")
                return False
        else:
            print("✗ Docker command failed")
            return False
    except FileNotFoundError:
        print("✗ Docker not found. Please install Docker Desktop.")
        return False
    except Exception as e:
        print(f"✗ Docker check failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\n" + "=" * 60)
    print("Testing CUDA/GPU...")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ CUDA not available (will use CPU - training will be slower)")
            return False
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Pac-Man RL Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test Docker
    docker_ok = test_docker()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if imports_ok and docker_ok:
        print("✓ Installation successful!")
        print("\nNext steps:")
        print("  1. Pull Docker image: docker pull ghcr.io/meta-pytorch/openenv-pacman-env:latest")
        print("  2. Start training: python train_pacman_docker_grpo_v2.py")
        print("  3. Check notebook: notebooks/Pacman-RL.ipynb")
        return 0
    else:
        print("✗ Installation incomplete. Please fix the errors above.")
        if not imports_ok:
            print("\n  Missing dependencies. Run: pip install -r requirements.txt")
        if not docker_ok:
            print("\n  Docker not available. Install Docker Desktop.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
