# CleanGallery

A specialized tool for cleaning and organizing image galleries using OpenAI's CLIP model and PyTorch.

## ⚠️ Important: Hardware & Installation Requirements

This project is optimized for NVIDIA GPU acceleration. To ensure the CLIP model runs efficiently, it is highly recommended to use the CUDA-enabled version of PyTorch.

### GPU Specifications
The development environment for this project used the following configuration:
* **CUDA Version:** 12.1
* **Compiler:** NVIDIA (R) Cuda compiler driver (V12.1.66)
* **Driver Release:** r12.1

### Pro tip
Use the following to "Verify GPU support". It’s a lifesaver for anyone else (or "future Me") trying to figure out why the script is running slowly on a new machine.
```bash
python -c "import torch; print(torch.cuda.is_available())"
