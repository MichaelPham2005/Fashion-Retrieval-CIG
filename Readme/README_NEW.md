# Generative Zero-Shot Composed Image Retrieval

<img width="2328" height="344" alt="image" src="https://github.com/user-attachments/assets/b4a3956c-4526-483e-8512-ba518a2b37d8" />

Zero-Shot Composed Image Retrieval vs. Pseudo Target-Aided Composed Image Retrieval. Conventional ZS-CIR methods map the image latent embedding into the token embedding space by textual inversion. The proposed Pseudo Target-Aided method provide additional information for composed embeddings from pseudo-target images.

## 📁 Cấu Trúc Project

```
ComposedImageGen/
│
├── 📄 Core Scripts
│   ├── extract_lincir_feat.py      # Trích xuất composed embeddings từ ảnh và captions
│   ├── test_CIG.py                 # Tạo ảnh composed sử dụng SDXL
│   ├── phi.py                      # Định nghĩa mô hình Phi network
│   ├── config.py                   # Quản lý paths và cấu hình tập trung
│   └── download_images.py          # Utility để download ảnh từ URLs
│
├── 📂 datasets/
│   ├── __init__.py
│   ├── dataset_utils.py            # Dataset classes (CIRRDataset, FashionIQDataset, ComposedEmbedsDataset)
│   │
│   └── FashionIQ/                  # FashionIQ dataset
│       ├── captions/               # Caption files cho mỗi category
│       │   ├── cap.dress.{train|val|test}.json
│       │   ├── cap.shirt.{train|val|test}.json
│       │   └── cap.toptee.{train|val|test}.json
│       │
│       ├── image_splits/           # Image split files
│       │   ├── split.dress.{train|val|test}.json
│       │   ├── split.shirt.{train|val|test}.json
│       │   └── split.toptee.{train|val|test}.json
│       │
│       └── images/                 # Image data và metadata
│           ├── convert.py          # Utility để convert URLs thành JSON
│           ├── dress.json          # URL database cho dress images
│           ├── shirt.json          # URL database cho shirt images
│           ├── toptee.json         # URL database cho toptee images
│           └── downloaded/         # [Tạo bởi download_images.py] Ảnh đã tải về
│               ├── dress/
│               ├── shirt/
│               └── toptee/
│
├── 📂 models/
│   ├── phi_best.pt                 # Pretrained Phi cho CLIP-ViT-L/14
│   └── phi_best_giga.pt           # Pretrained Phi cho CLIP-Giga
│
├── 📂 outputs/                     # [Tạo khi chạy] Kết quả output
│   ├── embeddings/                 # Composed embeddings (.pt files)
│   └── generated_images/           # Generated images từ SDXL
│
├── 📂 SEARLE_CIG/                  # SEARLE baseline với CIG
│   └── src/
│       ├── data_utils.py
│       ├── datasets.py
│       ├── encode_with_pseudo_tokens*.py
│       ├── generate_test_submission.py
│       ├── gpt_phrases_generation.py
│       ├── image_concepts_association.py
│       ├── oti_inversion.py
│       ├── phi.py
│       ├── train_phi.py
│       ├── utils_feat.py
│       └── validate.py
│
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # Documentation này

```

## 🔄 Data Flow Pipeline

### 1️⃣ Data Preparation

```
URLs (JSON files) → Download Script → Local Images → Dataset Loaders
```

### 2️⃣ Feature Extraction

```
Images + Captions → CLIP Models → Phi Networks → Composed Embeddings (.pt)
```

### 3️⃣ Image Generation

```
Composed Embeddings → SDXL Pipeline → Generated Images
```

### 4️⃣ Evaluation

```
Generated Images → SEARLE Baseline → Retrieval Metrics
```

## 📊 Dataset Details

### FashionIQ Dataset Structure

**Metadata Files (images/):**

```json
// dress.json, shirt.json, toptee.json
[
  {
    "asin": "B00014NGF6",
    "url": "http://ecx.images-amazon.com/images/I/51AERD8SM8L._SY445_.jpg"
  }
]
```

**Caption Files (captions/):**

```json
// cap.{category}.{split}.json
[
  {
    "target": "B008BHCT58",
    "candidate": "B003FGW7MK",
    "captions": ["is solid black with no sleeves", "is black with straps"]
  }
]
```

**Image Split Files (image_splits/):**

```json
// split.{category}.{split}.json
[
    "B00014NGF6",
    "B0006HJ8FU",
    ...
]
```

### CIRR Dataset Structure

```
cirr/
├── captions/
│   └── cap.rc2.{split}.json
├── test1/                    # Test images
│   └── {image_id}.png
├── dev/                      # Development images
│   └── {image_id}.png
└── ...
```

## 🚀 Getting Started

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Yêu cầu chính:**

- Python 3.8+
- PyTorch + CUDA
- transformers, diffusers (HuggingFace)
- CLIP (OpenAI)
- PIL, opencv, albumentations

### 2. Download Pre-trained Weights

Download và đặt vào thư mục `models/`:

- `phi_best.pt` - Phi model cho CLIP-ViT-L/14
- `phi_best_giga.pt` - Phi model cho CLIP-Giga

Link: [Google Drive](https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA?usp=drive_link)

### 3. Chuẩn Bị Dữ Liệu

#### Option A: Download Images từ URLs (FashionIQ)

```bash
python download_images.py --dataset fashioniq --categories dress shirt toptee
```

Script sẽ:

- Đọc URLs từ `datasets/FashionIQ/images/{category}.json`
- Download ảnh về `datasets/FashionIQ/images/downloaded/{category}/`
- Xử lý lỗi và retry khi cần

#### Option B: Sử dụng CIRR Dataset

Download CIRR dataset theo hướng dẫn [tại đây](https://github.com/miccunifi/SEARLE/tree/main#data-preparation) và đặt vào thư mục `datasets/CIRR/`

### 4. Cấu Hình Paths

Chỉnh sửa `config.py` hoặc truyền arguments khi chạy:

```python
# config.py
DATASET_PATHS = {
    'cirr': './datasets/CIRR',
    'fashioniq': './datasets/FashionIQ'
}

MODEL_PATHS = {
    'phi_vit': './models/phi_best.pt',
    'phi_giga': './models/phi_best_giga.pt',
    'sdxl_checkpoint': './models/sdxl_checkpoint/'
}

OUTPUT_PATHS = {
    'embeddings': './outputs/embeddings/',
    'generated_images': './outputs/generated_images/'
}
```

## 🎯 Usage

### Step 1: Trích Xuất Composed Embeddings

**Cho CIRR Dataset:**

```bash
python extract_lincir_feat.py \
    --dataset cirr \
    --text_embeddings_dir ./outputs/embeddings/cirr_test/
```

**Cho FashionIQ Dataset:**

```bash
python extract_lincir_feat.py \
    --dataset fashioniq \
    --text_embeddings_dir ./outputs/embeddings/fashioniq_test/
```

**Output:**

- Tạo file `.pt` cho mỗi pair trong `text_embeddings_dir/`
- Mỗi file chứa: `conditioning`, `conditioning2`, `pooled`, `pooled2`

### Step 2: Tạo Composed Images với SDXL

```bash
python test_CIG.py \
    --text_embeddings_dir ./outputs/embeddings/cirr_test/ \
    --dataset_dir ./datasets/CIRR \
    --model_path ./models/sdxl_checkpoint/ \
    --save_path ./outputs/generated_images/cirr/ \
    --batch_size 4 \
    --height 512 \
    --width 512 \
    --steps 50 \
    --seed 1600
```

**Parameters:**

- `--text_embeddings_dir`: Thư mục chứa embeddings từ Step 1
- `--dataset_dir`: Thư mục dataset gốc
- `--model_path`: Path đến SDXL checkpoint
- `--save_path`: Nơi lưu ảnh generated
- `--brightness_thresh`: Ngưỡng brightness để filter ảnh
- `--max_retries`: Số lần retry nếu ảnh quá tối

### Step 3: Evaluation với SEARLE

```bash
cd SEARLE_CIG
python src/generate_test_submission.py \
    --submission-name cirr_sdxl_b32 \
    --eval-type searle \
    --dataset cirr \
    --dataset-path ../datasets/CIRR \
    --generated-image-dir ../outputs/generated_images/cirr/
```

## 🧩 Components Chi Tiết

### 1. `phi.py` - Phi Network

```python
class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Chuyển đổi visual features thành pseudo-token embeddings.

    Architecture: Linear → GELU → Dropout → Linear → GELU → Dropout → Linear
    """
```

### 2. `extract_lincir_feat.py` - Feature Extraction

**Chức năng:**

- Load CLIP models (ViT-L/14 và Giga)
- Load Phi networks
- Xử lý reference images + relative captions
- Tạo composed embeddings cho SDXL

**Models được sử dụng:**

- `openai/clip-vit-large-patch14`
- `Geonmo/CLIP-Giga-config-fixed`

### 3. `test_CIG.py` - Image Generation

**Chức năng:**

- Load SDXL pipeline
- Batch processing embeddings
- Generate images với retry mechanism
- Filter theo brightness threshold

### 4. `dataset_utils.py` - Dataset Loaders

#### CIRRDataset

```python
Returns:
    {
        'reference_image': Tensor,
        'relative_caption': str,
        'pairid': str
    }
```

#### FashionIQDataset

```python
Returns:
    {
        'reference_image': Tensor,
        'relative_caption': str
    }
```

#### ComposedEmbedsDataset

```python
Returns:
    {
        'pairid': str,
        'prompt_embeds': Tensor [seq_len, d1+d2],
        'pooled2': Tensor [d2]
    }
```

## ⚠️ Lưu Ý Quan Trọng

### 1. Image URLs vs Local Files

- **FashionIQ**: Mặc định chứa URLs, cần download về local
- **CIRR**: Dataset có sẵn local images
- Code đã được update để hỗ trợ cả 2 modes

### 2. GPU Memory Requirements

- CLIP models: ~2GB VRAM
- SDXL pipeline: ~8GB VRAM
- Tổng khuyến nghị: GPU với ít nhất 12GB VRAM

### 3. Storage Requirements

- FashionIQ images: ~50GB (sau khi download)
- CIRR images: ~30GB
- Generated images: Depends on test set size

### 4. Hardcoded Paths

Các paths mặc định cần được update:

```python
# ❌ Cũ
dataset_path = "/path/to/cirr/dataset"

# ✅ Mới
dataset_path = args.dataset_path or "./datasets/CIRR"
```

## 🔧 Troubleshooting

### Lỗi: "Cannot load image from URL"

**Giải pháp:** Run `download_images.py` để tải ảnh về local trước

### Lỗi: "CUDA out of memory"

**Giải pháp:**

- Giảm `--batch_size`
- Dùng `torch_dtype=torch.float16`
- Close các process khác sử dụng GPU

### Lỗi: "Pretrained model not found"

**Giải pháp:** Download models và đặt đúng vào thư mục `models/`

### Ảnh generated quá tối

**Giải pháp:**

- Tăng `--max_retries`
- Giảm `--brightness_thresh`
- Thử seeds khác nhau

## 📈 Performance Tips

1. **Parallel Processing:** Tăng `--num_workers` trong DataLoader
2. **Batch Size:** Tăng batch_size nếu GPU memory đủ
3. **Mixed Precision:** Dùng `torch.float16` cho inference
4. **Caching:** CLIP models cache vào `./cache/` tự động

## 🔥 Updates

- [x] Pretrained weights
- [x] Inference code
- [x] Updated README với cấu trúc chi tiết
- [x] Download utility cho images
- [x] Fixed hardcoded paths
- [ ] Support more benchmarks and baselines
- [ ] Train code

## 📚 Citation

```bibtex
@inproceedings{wang2025CIG,
  title={Generative zero-shot composed image retrieval},
  author={Wang, Lan and Ao, Wei and Boddeti, Vishnu Naresh and Lim, Sernam},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

## 🙏 Acknowledgements

This project builds upon the following repositories:

- [SEARLE](https://github.com/miccunifi/SEARLE/tree/main)
- [lincir](https://github.com/navervision/lincir)

I am grateful to the authors and contributors of these projects for making their work available to the community.
