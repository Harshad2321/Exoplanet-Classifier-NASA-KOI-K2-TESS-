#!/usr/bin/env python3
"""
GPU Detection and Configuration Check for RTX 4060
This script verifies that both PyTorch and TensorFlow can detect and use your GPU.
"""

import sys
import os

def check_pytorch_gpu():
    """Check PyTorch GPU availability and configuration"""
    print("=" * 60)
    print("🔍 PYTORCH GPU DETECTION")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Test tensor operations on GPU
            print("\n🧪 Testing PyTorch GPU Operations:")
            device = torch.device("cuda")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print(f"✅ Matrix multiplication on GPU successful: {z.shape}")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            
        else:
            print("❌ CUDA not available. Possible issues:")
            print("  - NVIDIA drivers not installed or outdated")
            print("  - CUDA toolkit not installed")
            print("  - PyTorch not installed with CUDA support")
            print("  - GPU not detected by system")
            
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        print(f"❌ PyTorch GPU check failed: {str(e)}")

def check_tensorflow_gpu():
    """Check TensorFlow GPU availability and configuration"""
    print("\n" + "=" * 60)
    print("🔍 TENSORFLOW GPU DETECTION")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU Devices Found: {len(gpus)}")
        
        if gpus:
            print("GPU Details:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
            # Configure memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Memory growth configured for all GPUs")
            except RuntimeError as e:
                print(f"⚠️  Memory growth config: {e}")
            
            # Test TensorFlow operations on GPU
            print("\n🧪 Testing TensorFlow GPU Operations:")
            with tf.device('/GPU:0'):
                x = tf.random.normal([1000, 1000])
                y = tf.random.normal([1000, 1000])
                z = tf.matmul(x, y)
                print(f"✅ Matrix multiplication on GPU successful: {z.shape}")
                
            # Check if TensorFlow was built with CUDA
            print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"GPU Support: {tf.test.is_gpu_available()}")
            
        else:
            print("❌ No GPU devices found. Possible issues:")
            print("  - NVIDIA drivers not installed or outdated")
            print("  - CUDA toolkit not compatible with TensorFlow version")
            print("  - TensorFlow-GPU not installed properly")
            
    except ImportError:
        print("❌ TensorFlow not installed. Run: pip install tensorflow")
    except Exception as e:
        print(f"❌ TensorFlow GPU check failed: {str(e)}")

def check_system_info():
    """Check system and CUDA information"""
    print("\n" + "=" * 60)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"NVIDIA Driver: {line.split('Driver Version: ')[1].split()[0]}")
                if 'RTX 4060' in line or 'GeForce' in line:
                    print(f"GPU Found: {line.strip()}")
        else:
            print("❌ nvidia-smi not found or failed")
    except:
        print("❌ Could not run nvidia-smi")

def main():
    """Run all GPU checks"""
    print("🚀 RTX 4060 GPU CONFIGURATION CHECK")
    print("For NASA Exoplanet Classification Project")
    print("=" * 60)
    
    check_system_info()
    check_pytorch_gpu()
    check_tensorflow_gpu()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    print("If GPUs are not detected:")
    print("1. Update NVIDIA drivers: https://www.nvidia.com/drivers/")
    print("2. Install CUDA 12.1: https://developer.nvidia.com/cuda-downloads")
    print("3. Install PyTorch with CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("4. Install TensorFlow with GPU support:")
    print("   pip install tensorflow[and-cuda]")
    
    print("\nIf everything works:")
    print("✅ Your RTX 4060 is ready for accelerated ML training!")
    print("🚀 Run your training scripts to see performance improvements!")

if __name__ == "__main__":
    main()