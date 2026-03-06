# Generative Zero-Shot Composed Image Retrieval

<img width="2328" height="344" alt="image" src="https://github.com/user-attachments/assets/b4a3956c-4526-483e-8512-ba518a2b37d8" />

Zero-Shot Composed Image Retrieval vs. Pseudo Target-Aided Composed Image Retrieval. Conventional ZS-CIR methods map the image latent embedding into the token embedding space by textual inversion. The proposed Pseudo Target-Aided method provides additional information for composed embeddings from pseudo-target images.

---

## 📋 System Requirements

### Minimum Hardware

- **GPU**: NVIDIA CUDA-compatible GPU (minimum **12GB VRAM**)
  - RTX 3060 12GB, RTX 4060 8GB+ or equivalent
  - Recommended: RTX 3090, A100, or T4 (on Colab)
- **RAM**: Minimum **16GB** RAM (recommended **32GB**)
- **Disk**: Minimum **50GB** for models and cache

### Software Requirements

- **Python**: 3.9 or 3.10+
- **CUDA**: 11.8+ (if running locally)
- **cuDNN**: 8.0+ (if running locally)

> **⚠️ Note**: CPU-only execution is extremely slow. A GPU is strongly recommended (local or cloud).

---

## 🚀 Getting Started

### **Option 1: Running on Google Colab (Recommended for users without GPU)**

#### Step 1: Prepare Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new Notebook or upload `colab_setup_fixed.ipynb` from this repo
3. Select Runtime: `Runtime` → `Change runtime type` → **GPU (T4)** or **GPU (A100)** if available

#### Step 2: Run Colab Setup

```python
# Cell 1: Clone repo and install dependencies
!git clone https://github.com/MichaelPham2005/Fashion-Retrieval-CIG.git
%cd Fashion-Retrieval-CIG
!pip install -r requirements.txt --quiet
```

```python
# Cell 2: Download models (run only once)
!python extract_database_features.py
# Or download from Google Drive if pre-prepared
```

```python
# Cell 3: Start Flask API Server
import subprocess
import os

# Install Ngrok (to expose server to the internet)
!pip install pyngrok --quiet
from pyngrok import ngrok

# Get authentication token from https://dashboard.ngrok.com/auth/your-authtoken
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# Run Flask API
exec(open('api_pseudo_target.py').read())
```

#### Step 3: Get Ngrok URL

After the server starts, copy the Ngrok URL (format: `https://xxxxx.ngrok.io`)

#### Step 4: Run Frontend on Local PC

1. Download `demo_website_interactive.html` from the repo
2. Open the file in your web browser
3. Enter the Ngrok URL in the "Server Configuration" field
4. Click "Test Connection" to verify

---

### **Option 2: Running on Local PC (Windows/Linux/Mac)**

#### Step 1: Set up environment

```bash
# Clone repo
git clone https://github.com/MichaelPham2005/Fashion-Retrieval-CIG.git
cd Fashion-Retrieval-CIG
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Download models and database embeddings

```bash
# Create database embeddings (if not already present)
python extract_database_features.py

# Or download from Google Drive
# https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA?usp=drive_link
```

#### Step 4: Start Flask API Server

```bash
python api_pseudo_target.py
```

Server will run at `http://localhost:5000`

#### Step 5: Open Frontend

1. Open `demo_website_interactive.html` in your web browser
2. Enter `http://localhost:5000` in the "Server Configuration" field
3. Click "Test Connection"

---

## 📂 Directory Structure

```
ComposedImageGen/
├── api_pseudo_target.py              # Flask API backend (runs on Colab/Local)
├── config.py                         # Configuration (model paths, hyperparameters)
├── demo_website_interactive.html     # Frontend HTML (runs in local browser)
├── colab_setup_fixed.ipynb          # Jupyter notebook for Colab
│
├── models/                           # Models directory (requires download)
│   ├── phi_best.pt                  # Phi network weights
│   ├── phi_best_giga.pt             # Phi network weights (large version)
│   └── checkpoint-20000-SDXL/       # SDXL fine-tuned weights
│
├── database_embeddings/              # Embeddings directory (requires creation)
│   └── fashioniq_database.pt        # Database embeddings
│
├── datasets/                         # Benchmark data
│   └── FashionIQ/                   # FashionIQ dataset
│
├── cache/                            # Cache for HuggingFace models
│   └── models--openai--clip-vit-large-patch14/
│
├── SEARLE_CIG/                      # Baseline implementations
│   └── src/
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚙️ Running Basic Scripts

### 1. Database Feature Extraction

```bash
# Create embeddings for all images in FashionIQ
python extract_database_features.py
```

**Output**: `database_embeddings/fashioniq_database.pt` (contains embeddings + URL mappings)

### 2. Test Composed Image Generation

```bash
# Generate pseudo-target image from reference image + text query
python test_CIG.py
```

### 3. Test with SEARLE Baseline

```bash
cd SEARLE_CIR
python src/generate_test_submission.py \
    --submission-name cirr_sdxl_b32 \
    --eval-type searle \
    --dataset cirr \
    --dataset-path /path/to/CIRR \
    --generated-image-dir /path/to/generated_images
```

---

## 🔧 Configuration

The `config.py` file contains important hyperparameters:

```python
# Image preprocessing
IMAGE_PREPROCESS = {
    'size': 224,
    'crop_size': 224,
    'image_mean': [0.48145466, 0.4578275, 0.40821073],
    'image_std': [0.26862954, 0.26130258, 0.27577711],
}

# Model paths
MODEL_PATHS = {
    'phi_vit': './models/phi_best.pt',
    'sdxl_checkpoint': './models/checkpoint-20000-SDXL/',
}

# SDXL generation parameters
SDXL_GENERATION = {
    'height': 512,
    'width': 512,
    'num_inference_steps': 50,
    'guidance_scale': 7.5,
    'brightness_threshold': 50,
}
```

Modify these values if you need to optimize for your hardware.

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"

- **Solution**: Reduce batch size or image resolution in `config.py`
- Or use Colab with A100 GPU (40GB VRAM)

### Error: "UNet not found"

- **Solution**: Check model path in `config.py`
- Download SDXL weights from [Google Drive](https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA?usp=drive_link)

### Error: "Database embeddings not found"

- **Solution**: Run `python extract_database_features.py` to create embeddings

### Colab server cannot connect from local

- **Solution**:
  1. Verify Ngrok auth token is entered correctly
  2. Check if firewall is blocking HTTPS
  3. Ensure `ngrok-skip-browser-warning: true` header is in frontend (already added)

---

## 📊 Expected Results

### Performance Metrics

| Step                | Local (RTX 3090) | Colab (T4)  | Colab (A100) |
| ------------------- | ---------------- | ----------- | ------------ |
| Trích CLIP features | 0.2s             | 0.5s        | 0.3s         |
| Phi prediction      | 0.05s            | 0.1s        | 0.05s        |
| SDXL generation     | 8-12s            | 15-20s      | 5-8s         |
| Database search     | 0.5s             | 0.5s        | 0.5s         |
| **Total**           | **~9-13s**       | **~16-21s** | **~6-9s**    |

### Interactive Refinement (Rocchio)

- **Speed**: **< 0.5s** (no need to regenerate image)
- **Accuracy**: 15-25% improvement over initial results

---

## 📚 References

- **SEARLE**: [https://github.com/miccunifi/SEARLE](https://github.com/miccunifi/SEARLE)
- **LINCIR**: [https://github.com/navervision/lincir](https://github.com/navervision/lincir)
- **Stable Diffusion XL**: [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **CLIP**: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

---

## 🔥 Updates

- [x] Pretrained weights
- [x] Inference code
- [x] **Colab deployment with Flask API** ✨
- [x] **Interactive Relevance Feedback (Rocchio Algorithm)** ✨
- [x] **Web-based Demo Interface** ✨
- [ ] Support more benchmarks and baselines
- [ ] Train code

---

## ✨ New Features

### 1. **Hybrid Architecture: Colab Backend + Local Frontend**

- Backend runs on Google Colab (free GPU)
- Frontend runs on local PC (HTML/JavaScript)
- Connection via Ngrok tunnel
- **Benefits**: No local GPU needed, saves power

### 2. **Pseudo-Target Generation Pipeline**

```
Input Image + Text Query
      ↓
Extract CLIP features
      ↓
Phi network prediction
      ↓
SDXL image generation
      ↓
Feature extraction + Database search
      ↓
Top-K results
```

### 3. **Interactive Relevance Feedback**

- After initial search, users can select images they like
- System uses **Rocchio Algorithm** to refine query
- No need to regenerate image (< 0.5s)
- Results improve continuously through iterations

**Rocchio Formula:**
$$Q_{refined} = \alpha \cdot mean(selected\_embeddings) + (1-\alpha) \cdot Q_{original}$$

---

## 🎥 Demo Video Script

To create a demo video, we've prepared a detailed script (see `Readme/DEMO_SCRIPT.md`).

**Video Content:**

1. **Introduction** (2 min): Pseudo-Target concept
2. **Demo 1** (3 min): Basic search (upload image + query)
3. **Demo 2** (3 min): Interactive Refinement feature
4. **Conclusion** (1 min)

---

## 🛠️ API Endpoints

### `POST /search`

Search for composed images with pseudo-target generation

**Request:**

```json
{
  "image": <file>,
  "query": "change to floral pattern, make it shorter",
  "top_k": 20,
  "height": 512,
  "width": 512,
  "steps": 50,
  "seed": 42
}
```

**Response:**

```json
{
  "results": [
    {"asin": "product_id", "url": "image_url", "score": 0.95},
    ...
  ],
  "pseudo_target_image": "base64_encoded_image",
  "generation_time": 12.5,
  "search_time": 0.3
}
```

### `POST /refine`

Refine results based on user-selected images

**Request:**

```json
{
  "selected_results": ["id_1", "id_2", "id_3"],
  "iteration": 2,
  "top_k": 20,
  "alpha": 0.7
}
```

**Response:**

```json
{
  "results": [...],
  "iteration": 2,
  "refined": true
}
```

### `GET /health`

Check server status

**Response:**

```json
{
  "status": "healthy",
  "device": "cuda",
  "database_size": 18544,
  "sdxl_loaded": true,
  "approach": "Pseudo-Target Generation"
}
```

---

## 📝 Dependencies Reference

File `requirements.txt` includes:

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.21.0
Pillow>=9.5.0
numpy>=1.24.0
Flask>=2.3.0
flask-cors>=4.0.0
pyngrok>=5.1.0
clip==1.0
einops>=0.7.0
omegaconf>=2.3.0
```

Update: Run `pip install -r requirements.txt --upgrade` to get the latest versions

---

## 🤝 Citation

```bibtex
@inproceedings{wang2025CIG,
  title={Generative zero-shot composed image retrieval},
  author={Wang, Lan and Ao, Wei and Boddeti, Vishnu Naresh and Lim, Sernam},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This project builds upon the following repositories:

- [SEARLE](https://github.com/miccunifi/SEARLE/tree/main)
- [lincir](https://github.com/navervision/lincir)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [CLIP](https://github.com/openai/CLIP)

I am grateful to the authors and contributors of these projects for making their work available to the community.

---

## 📞 Support & Issues

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review log files in the `logs/` directory
3. Open an issue on GitHub with error details and hardware specifications
