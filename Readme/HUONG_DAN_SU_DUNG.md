# 🚀 Hướng Dẫn Sử Dụng Nhanh - ComposedImageGen

## 📋 Tổng Quan

Project này thực hiện **Composed Image Retrieval** - tạo ảnh mới từ ảnh tham chiếu + mô tả văn bản.

**Input:** Ảnh gốc + "Tôi muốn ảnh này nhưng màu đỏ hơn"
**Output:** Ảnh mới phù hợp với mô tả

---

## ✅ Những Gì Đã Sửa

### Vấn đề ban đầu:

- ❌ Dataset chỉ có URL, không có ảnh thật
- ❌ Đường dẫn hardcoded không tồn tại
- ❌ Không có hướng dẫn rõ ràng
- ❌ Code không linh hoạt

### Giải pháp:

- ✅ Tạo script download ảnh từ URL
- ✅ Sửa dataset loader hỗ trợ cả URL và local
- ✅ Tạo config.py quản lý paths tập trung
- ✅ Sửa code linh hoạt với arguments
- ✅ Viết README chi tiết
- ✅ Thêm error handling và logging

---

## 📂 Cấu Trúc Mới

```
ComposedImageGen/
├── config.py                    # ⭐ MỚI - Quản lý paths & settings
├── download_images.py           # ⭐ MỚI - Download ảnh từ URLs
├── extract_lincir_feat.py       # ✏️ ĐÃ SỬA - Extract features
├── test_CIG.py                  # ✏️ ĐÃ SỬA - Generate images
├── phi.py                       # ✓ Không đổi
├── requirements.txt             # ✓ Không đổi
├── README_NEW.md                # ⭐ MỚI - Docs chi tiết
├── CHANGES_SUMMARY.md           # ⭐ MỚI - Tổng hợp thay đổi
│
├── datasets/
│   ├── dataset_utils.py         # ✏️ ĐÃ SỬA - Support URL + local
│   └── FashionIQ/
│       ├── captions/            # Caption files sẵn có
│       ├── image_splits/        # Split files sẵn có
│       └── images/
│           ├── dress.json       # URLs sẵn có
│           ├── shirt.json       # URLs sẵn có
│           ├── toptee.json      # URLs sẵn có
│           └── downloaded/      # Sẽ tạo khi download
│               ├── dress/
│               ├── shirt/
│               └── toptee/
│
├── models/                      # Cần download pretrained models
│   ├── phi_best.pt
│   ├── phi_best_giga.pt
│   └── sdxl_checkpoint/
│
└── outputs/                     # Tự động tạo khi chạy
    ├── embeddings/
    └── generated_images/
```

---

## 🎯 Workflow Hoàn Chỉnh

### 0️⃣ Setup môi trường

```bash
# Cài packages
pip install -r requirements.txt

# Kiểm tra config
python config.py
```

**Output mẫu:**

```
======================================================================
ComposedImageGen Configuration
======================================================================

📂 Dataset Paths:
  ❌ cirr: ./datasets/CIRR
  ✅ fashioniq: ./datasets/FashionIQ

🤖 Model Paths:
  ❌ phi_vit: ./models/phi_best.pt
  ❌ phi_giga: ./models/phi_best_giga.pt

⚠️  Model not found: phi_vit at ./models/phi_best.pt
    Please download models from: https://drive.google.com/...
```

### 1️⃣ Download pretrained models

Tải về từ [Google Drive](https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA) và đặt vào:

- `models/phi_best.pt`
- `models/phi_best_giga.pt`

### 2️⃣ Download images (FashionIQ)

```bash
# Download tất cả categories
python download_images.py --dataset fashioniq --categories dress shirt toptee

# Hoặc download từng category
python download_images.py --dataset fashioniq --categories dress
```

**Output mẫu:**

```
======================================================================
📥 FashionIQ Image Downloader
======================================================================

📂 Loading dress data from ./datasets/FashionIQ/images/dress.json
✅ Found 25000 images in dress
📥 Downloading 25000 images for dress...
Downloading dress: 100%|████████| 25000/25000 [15:30<00:00, 26.88img/s]

✅ DRESS Complete:
   - Total: 25000
   - Success: 24850
   - Failed: 150
   - Skipped: 0
```

**Lưu ý:**

- Download tốn thời gian (10-20 phút/category)
- Một số URLs có thể fail → sẽ log vào `failed_downloads_*.json`
- Có thể chạy lại script để retry failed images

### 3️⃣ Extract features từ images + captions

```bash
# Cho FashionIQ test set
python extract_lincir_feat.py \
    --dataset fashioniq \
    --split test \
    --batch_size 4

# Với custom paths
python extract_lincir_feat.py \
    --dataset fashioniq \
    --split test \
    --dataset_path ./datasets/FashionIQ \
    --text_embeddings_dir ./my_embeddings \
    --batch_size 8 \
    --num_workers 4
```

**Output mẫu:**

```
======================================================================
LinCIR Feature Extraction
======================================================================
Dataset: fashioniq
Dataset Path: ./datasets/FashionIQ
Split: test
Output Directory: ./outputs/embeddings/fashioniq_test
Batch Size: 4
Device: cuda
======================================================================

📥 Loading CLIP models...
✅ CLIP models loaded

📥 Loading Phi models...
✅ Phi models loaded

📂 Loading dataset...
✅ Dataset loaded: 3000 samples

🚀 Extracting features...
Extracting features (fashioniq): 100%|████| 750/750 [05:23<00:00, 2.32batch/s]

======================================================================
✅ Feature extraction complete!
Total processed: 3000
Errors: 0
Output directory: ./outputs/embeddings/fashioniq_test
======================================================================
```

**Output files:**

- Mỗi sample → 1 file `.pt`
- Chứa: `conditioning`, `conditioning2`, `pooled`, `pooled2`
- Location: `outputs/embeddings/{dataset}_{split}/{id}.pt`

### 4️⃣ Generate images với SDXL

```bash
# Basic usage
python test_CIG.py \
    --dataset fashioniq \
    --split test \
    --batch_size 4 \
    --steps 50

# Với custom settings
python test_CIG.py \
    --dataset fashioniq \
    --split test \
    --text_embeddings_dir ./outputs/embeddings/fashioniq_test \
    --save_path ./my_generated_images \
    --batch_size 2 \
    --height 512 \
    --width 512 \
    --steps 30 \
    --brightness_thresh 10.0 \
    --max_retries 5
```

**Output mẫu:**

```
======================================================================
Composed Image Generation with SDXL
======================================================================
Dataset: fashioniq
Embeddings Directory: ./outputs/embeddings/fashioniq_test
Output Directory: ./outputs/generated_images/fashioniq_test
Batch Size: 4
Image Size: 512x512
Inference Steps: 50
Device: cuda
======================================================================

📂 Loading dataset...
✅ Dataset loaded: 3000 samples

📥 Loading SDXL models...
✅ SDXL pipeline loaded

🚀 Generating images...
Total samples to process: 3000
Generating images: 100%|████| 750/750 [45:20<00:00, 3.63s/batch]

======================================================================
✅ Image generation complete!
Generated: 3000
Skipped (already exist): 0
Output directory: ./outputs/generated_images/fashioniq_test
======================================================================
```

**Features:**

- Brightness filtering (retry nếu ảnh quá tối)
- Skip already generated images (có thể resume)
- Batch processing for efficiency

---

## 💡 Tips & Tricks

### Memory Management

```bash
# Nếu GPU out of memory:
python extract_lincir_feat.py --batch_size 2  # giảm batch size
python test_CIG.py --batch_size 1 --steps 30   # giảm steps
```

### Test trên sample nhỏ

```bash
# Test trên 10 samples đầu tiên để debug
# Sửa trong code: dataset[:10]
```

### Resume từ checkpoint

```bash
# Script tự động skip images đã generate
# Chỉ cần chạy lại lệnh cũ
python test_CIG.py --dataset fashioniq --split test
```

### Check progress

```bash
# Đếm số embeddings đã extract
ls outputs/embeddings/fashioniq_test/*.pt | wc -l

# Đếm số images đã generate
ls outputs/generated_images/fashioniq_test/*.png | wc -l
```

---

## ⚠️ Troubleshooting

### Lỗi: "Image not found"

**Nguyên nhân:** Chưa download images hoặc URL fail
**Giải pháp:**

```bash
python download_images.py --dataset fashioniq --categories dress
```

### Lỗi: "Model not found"

**Nguyên nhân:** Chưa download pretrained models
**Giải pháp:** Download từ Google Drive và đặt vào `models/`

### Lỗi: "CUDA out of memory"

**Giải pháp:**

```bash
# Giảm batch size
python extract_lincir_feat.py --batch_size 1
python test_CIG.py --batch_size 1

# Hoặc dùng CPU (chậm hơn nhiều)
python extract_lincir_feat.py --device cpu
```

### Lỗi: "Caption file not found"

**Nguyên nhân:** Dataset path không đúng
**Giải pháp:**

```bash
# Check path
python config.py

# Hoặc dùng custom path
python extract_lincir_feat.py --dataset_path /correct/path/to/FashionIQ
```

### Images quá tối

**Giải pháp:**

```bash
# Tăng retries
python test_CIG.py --max_retries 10

# Giảm threshold
python test_CIG.py --brightness_thresh 5.0

# Đổi seed
python test_CIG.py --seed 2024
```

---

## 📊 Expected Results

### Dataset sizes:

- **FashionIQ dress:** ~25,000 images
- **FashionIQ shirt:** ~11,000 images
- **FashionIQ toptee:** ~12,000 images
- **Total:** ~48,000 images

### Processing time (với GPU RTX 3090):

- **Download:** ~15 min/category
- **Extract features:** ~5-10 min/1000 samples
- **Generate images:** ~30-60 min/1000 samples (50 steps)

### Storage requirements:

- **Downloaded images:** ~50GB
- **Embeddings:** ~500MB/1000 samples
- **Generated images:** ~2GB/1000 samples

---

## 🎓 Giải Thích Kỹ Thuật

### Pipeline chi tiết:

1. **Input Processing:**

   - Reference image → CLIP Vision Encoder → Image features
   - Text caption → Tokenization

2. **Feature Extraction:**

   - Image features → Phi Network → Pseudo tokens
   - Tokens + Caption → CLIP Text Encoder → Composed embeddings

3. **Image Generation:**

   - Composed embeddings → SDXL UNet → Latent representation
   - Latent → VAE Decoder → Generated image

4. **Quality Control:**
   - Brightness check
   - Retry with different seeds if needed

### Models được dùng:

- **CLIP ViT-L/14:** Extract visual features (768-dim)
- **CLIP Giga:** Alternative CLIP model (1280-dim)
- **Phi Networks:** Transform visual → text embeddings
- **SDXL:** Stable Diffusion XL for image generation

---

## 📚 Tài Liệu Tham Khảo

- **README_NEW.md:** Documentation đầy đủ
- **CHANGES_SUMMARY.md:** Chi tiết các thay đổi
- **config.py:** Tất cả settings và paths
- **Original paper:** Check citations trong README

---

## ✅ Checklist Trước Khi Chạy

- [ ] Đã cài đặt requirements.txt
- [ ] Đã download pretrained models (phi_best.pt, phi_best_giga.pt)
- [ ] Đã chạy `python config.py` để validate
- [ ] Đã download images (cho FashionIQ)
- [ ] GPU có đủ VRAM (ít nhất 12GB recommended)
- [ ] Đủ disk space (~100GB cho full pipeline)

---

Chúc bạn thành công! 🎉

Nếu có vấn đề, check CHANGES_SUMMARY.md hoặc README_NEW.md để biết thêm chi tiết.
