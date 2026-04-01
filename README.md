# WebUI for ML-Sharp (3DGS)

[![GitHub Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/francescofugazzi)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/franzipol)

A seamless Pinokio-ready Web UI for **Apple's ML-Sharp**, allowing you to generate 3D Gaussian Splatting (3DGS) models from a single image with high efficiency.

## 🚀 Features

- **Single Click Install**: Fully automated dependency management via Pinokio.
- **Efficient Generation**: Optimized 3D PLY and preview video generation.
- **Standard PLY Export**: Produces externally-compatible 3DGS PLY files that work in SuperSplat, Luma, Antimatter15, and all standard viewers.
- **Intuitive Web UI**: 
    - **New Job**: Upload an image and get your 3D assets immediately.
    - **Result History**: Manage, download, and review your previous generations.
- **Smart Management**: Automatic cache cleanup to keep your installation lean.
- **High Quality**: Supports 3DGS PLY format and optional depth video rendering.

## 📦 Installation

1. Open **Pinokio**.
2. Click on **Discover** or **Download**.
3. Paste the URL of this repository.
4. Click **Install**.
5. Once finished, click **Start**.

## 🛠 Usage

1. **Upload**: Drag and drop an image into the "New Job" tab.
2. **Configure**: Select "Generate Video Immediately" if you have an NVIDIA GPU.
3. **Run**: Click "Start Generation".
4. **Download**: Once finished, download the `_standard.ply` model or videos from the file list.
5. **History**: Access the "Result History" tab to manage your collection.

## 🔧 PLY Compatibility Fix

This fork includes an automatic patch for ML-Sharp's PLY export format. Apple's original implementation writes a custom multi-element PLY (with extrinsic, intrinsic, image_size, and other metadata elements) that most external 3DGS viewers cannot parse. 

**What this fix does:**

- Automatically generates a `*_standard.ply` alongside the original for every export
- The standard PLY contains only the `vertex` element with the 17 standard properties (xyz, normals, SH, opacity, scales, rotations)
- The download link in the UI points to the standard PLY by default
- The built-in 3D viewer continues to use the original format (converted for Gradio)
- The patch is applied during installation and is idempotent (safe to re-run)

## 🖥 Requirements

- **Pinokio**: [https://pinokio.computer](https://pinokio.computer)
- **Apple Silicon (Mac)**: Highly recommended and extremely fast for 3D generation using MPS.
    - **Requirement**: Minimum **16GB Unified Memory** is required for stable operation.
- **NVIDIA GPU**: Required for the optional video rendering (CUDA), also provides high-speed generation.
    - **Performance Note**: Typical usage requires approximately **9GB VRAM**. A GPU with **10GB+** is recommended for best performance. Smaller VRAM capacities are supported via virtual memory/shared RAM (slower).
- **CPU / Other GPUs**: Supported via a generic installation path (performance may vary).
- **Windows/Linux/Mac**: Fully cross-platform.

## 📜 Credits

- **SHARP (ML-Sharp)**: Sharp Monocular View Synthesis in Less Than a Second. [GitHub Repository](https://github.com/apple/ml-sharp)
- **3DGS**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering.

---
Developed for Pinokio Ecosystem.
