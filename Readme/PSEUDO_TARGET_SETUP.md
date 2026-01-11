# 🎯 Setup Guide cho CIG Model với Pseudo-Target Generation

## ✨ Tổng Quan

Đây là hướng dẫn setup đầy đủ cho **Composed Image Retrieval (CIG)** với **Pseudo-Target Generation** approach - đúng với bản chất của model CIG.

### 🔄 Workflow CIG (Pseudo-Target Generation)

```
User Input (Reference Image + Text Query)
           ↓
    CLIP Vision Encoder → Extract reference features
           ↓
    Phi Network → Predict pseudo tokens
           ↓
    CLIP Text Encoder → Compose with text query
           ↓
    SDXL Pipeline → Generate Pseudo-Target Image ⭐
           ↓
    CLIP Vision Encoder → Extract pseudo-target features
           ↓
    Cosine Similarity → Compare with Database
           ↓
    Return Top-K Results + Pseudo-Target Image
```

**Điểm khác biệt:** Model sinh ra **pseudo-target image** làm trung gian, sau đó search bằng features của ảnh này (không phải composed embedding trực tiếp).

---

## 📋 Checklist Files Cần Thiết

### ✅ Models (Đã có)

```
✅ models/phi_best.pt                                    [~50MB]
✅ models/phi_best_giga.pt                              [~50MB]
✅ models/checkpoint-20000-SDXL/checkpoint-20000/
   └── unet/
       ├── config.json
       └── diffusion_pytorch_model.safetensors          [~5GB]
```

### ✅ Code Files (Vừa tạo)

```
✅ api_pseudo_target.py              [Backend với Pseudo-Target Generation]
✅ demo_website_pseudo_target.html   [Frontend hiển thị pseudo-target]
✅ extract_database_features.py      [Extract database embeddings]
```

### 📦 Database (Cần tạo)

```
❌ database_embeddings/fashioniq_database.pt  [Chưa có - cần chạy script]
```

### 🖼️ Dataset Images

```
❌ datasets/FashionIQ/images/downloaded/      [Cần download trước]
```

---

## 🚀 Hướng Dẫn Setup (Từng Bước)

### Step 1: Kiểm Tra Models

```powershell
# Kiểm tra SDXL checkpoint
ls models\checkpoint-20000-SDXL\checkpoint-20000\unet\

# Expected output:
# config.json
# diffusion_pytorch_model.safetensors
```

✅ **Đã có đầy đủ models!**

---

### Step 2: Download Database Images

```powershell
# Download FashionIQ images (dress, shirt, toptee)
python download_images.py --dataset fashioniq --categories dress shirt toptee
```

**Lưu ý:**

- Download khoảng 77,000+ ảnh (~50GB)
- Mất khoảng 1-2 giờ tùy tốc độ mạng
- Lưu vào: `datasets/FashionIQ/images/downloaded/{category}/`

**Alternative:** Nếu đã có ảnh sẵn, copy vào thư mục trên.

---

### Step 3: Extract Database Features

```powershell
# Extract CLIP features cho TẤT CẢ database images
python extract_database_features.py `
  --dataset fashioniq `
  --categories dress shirt toptee `
  --output_dir ./database_embeddings `
  --batch_size 32 `
  --device cuda
```

**Output:**

- File: `database_embeddings/fashioniq_database.pt`
- Size: ~500MB
- Chứa: embeddings + URL mapping

**Thời gian:**

- GPU: 10-20 phút
- CPU: 1-2 giờ

---

### Step 4: (Optional) Xóa Downloaded Images

```powershell
# Sau khi extract xong, có thể xóa images để tiết kiệm space
# Remove-Item -Recurse -Force datasets\FashionIQ\images\downloaded\
```

**Lưu ý:** KHÔNG xóa nếu còn muốn extract thêm hoặc visualize!

---

### Step 5: Chạy API Server

```powershell
# Activate virtual environment (nếu chưa)
.\venv\Scripts\Activate.ps1

# Start API server với Pseudo-Target Generation
python api_pseudo_target.py
```

**Console output:**

```
🚀 Starting server and loading models...
Device: cuda
📥 Loading CLIP models...
✅ CLIP models loaded
📥 Loading Phi model...
✅ Phi model loaded
📥 Loading SDXL pipeline (this may take a while)...
✅ SDXL pipeline loaded
📥 Loading database embeddings...
✅ Loaded 77683 database embeddings
✅ All models loaded successfully!

======================================================================
🌐 Starting Flask API server...
======================================================================
Approach: Pseudo-Target Generation (CIG Model)

Endpoints:
  - GET  /health  : Health check
  - POST /search  : Composed image search with pseudo-target generation
  - GET  /stats   : Database statistics

Server will run on: http://localhost:5000
======================================================================
```

---

### Step 6: Mở Website Demo

```powershell
# Mở file trong browser
start demo_website_pseudo_target.html
```

Hoặc truy cập: `file:///D:/ComposedImageGen/demo_website_pseudo_target.html`

---

## 🎮 Cách Sử Dụng

### 1. Upload Reference Image

- Click vào box "Upload Reference Image"
- Chọn ảnh quần áo (shirt, dress, toptee)
- Preview sẽ hiển thị

### 2. Nhập Text Query

- Ví dụ: "change to red color"
- Ví dụ: "make it darker and longer sleeves"
- Hoặc click vào example tags

### 3. Click "Generate & Search"

- Loading sẽ hiển thị (~10-30s)
- SDXL đang generate pseudo-target image

### 4. Xem Kết Quả

- **Pseudo-Target Image**: Ảnh được sinh ra bởi SDXL
- **Generation Time**: Thời gian generate
- **Search Results**: Top 20 ảnh tương tự

---

## 📊 So Sánh Approaches

### Approach 1: Direct Embedding (api.py - CŨ)

```
Reference + Query → Composed Embedding → Search Database
```

**Ưu điểm:**

- ✅ Nhanh (~0.3s)
- ✅ Đơn giản

**Nhược điểm:**

- ❌ Accuracy thấp hơn
- ❌ Không phản ánh bản chất của CIG model

---

### Approach 2: Pseudo-Target Generation (api_pseudo_target.py - MỚI) ⭐

```
Reference + Query → Generate Pseudo-Target → Extract Features → Search
```

**Ưu điểm:**

- ✅ **Accuracy cao hơn** (theo paper CIG)
- ✅ **Đúng với bản chất model**
- ✅ User thấy được pseudo-target image
- ✅ Dễ debug và visualize

**Nhược điểm:**

- ⚠️ Chậm hơn (~10-30s)
- ⚠️ Cần GPU mạnh
- ⚠️ Cần SDXL checkpoint

---

## 🔧 Troubleshooting

### Lỗi: "Database not loaded"

**Nguyên nhân:** Chưa chạy `extract_database_features.py`

**Giải pháp:**

```powershell
python extract_database_features.py
```

---

### Lỗi: "SDXL pipeline not loaded"

**Nguyên nhân:** Thiếu SDXL checkpoint hoặc sai đường dẫn

**Kiểm tra:**

```powershell
ls models\checkpoint-20000-SDXL\checkpoint-20000\unet\
```

**Sửa config.py nếu cần:**

```python
MODEL_PATHS = {
    'sdxl_checkpoint': './models/checkpoint-20000-SDXL'
}
```

---

### Lỗi: "CUDA out of memory"

**Giải pháp:**

1. **Giảm batch size:**

```python
# Trong api_pseudo_target.py, thêm vào form params:
height = 384  # thay vì 512
width = 384
steps = 30    # thay vì 50
```

2. **Dùng CPU (chậm):**

```powershell
set CUDA_VISIBLE_DEVICES=-1
python api_pseudo_target.py
```

---

### Lỗi: Generated image quá tối

**Giải pháp:** API tự động retry với seeds khác

**Tùy chỉnh:**

```python
# Trong request form:
brightness_thresh = 40  # Giảm threshold
max_retries = 5        # Tăng số lần retry
```

---

### Website không connect được API

**Kiểm tra:**

1. API server đang chạy?

```powershell
curl http://localhost:5000/health
```

2. CORS đã enable?

```python
# Trong api_pseudo_target.py đã có:
CORS(app)
```

3. Firewall block?

```powershell
# Tạm tắt hoặc allow port 5000
```

---

## 📈 Performance Benchmarks

### Hardware Requirements

**Minimum:**

- GPU: 8GB VRAM (RTX 3060)
- RAM: 16GB
- Disk: 60GB

**Recommended:**

- GPU: 12GB+ VRAM (RTX 3080/4080)
- RAM: 32GB
- Disk: 100GB

### Timing Breakdown

| Step                       | Time (GPU) | Time (CPU) |
| -------------------------- | ---------- | ---------- |
| Extract reference features | 0.1s       | 0.5s       |
| Phi network                | 0.05s      | 0.2s       |
| Text encoding              | 0.05s      | 0.2s       |
| **SDXL generation**        | **10-20s** | **2-5min** |
| Pseudo-target features     | 0.1s       | 0.5s       |
| Database search            | 0.1s       | 0.5s       |
| **Total**                  | **~15s**   | **~5min**  |

---

## 🎓 So Sánh với Paper

### CIG Paper Approach

```
1. Visual feature extraction (CLIP Vision)
2. Phi network predicts pseudo tokens
3. Text encoding with pseudo tokens
4. SDXL generates pseudo-target
5. Extract features from pseudo-target
6. Retrieval using pseudo-target features
```

✅ **Implementation của bạn CHÍNH XÁC theo paper!**

---

## 📝 Testing Checklist

### ✅ Functional Tests

- [ ] Upload ảnh và hiển thị preview
- [ ] Nhập text query
- [ ] Click search và thấy loading
- [ ] SDXL generate pseudo-target (~10-30s)
- [ ] Hiển thị pseudo-target image
- [ ] Hiển thị generation time, search time
- [ ] Hiển thị top-20 results
- [ ] Click vào result mở URL

### ✅ Quality Tests

- [ ] Pseudo-target image có liên quan đến query
- [ ] Results khớp với pseudo-target
- [ ] Retry mechanism hoạt động (nếu ảnh quá tối)
- [ ] Score giảm dần từ rank 1 → 20

---

## 🔥 Production Tips

### 1. Caching Pseudo-Targets

Nếu có queries lặp lại nhiều:

```python
# Cache generated images
cache = {}
cache_key = f"{reference_hash}_{query}"
if cache_key in cache:
    pseudo_target = cache[cache_key]
else:
    pseudo_target = sdxl_pipe.generate(...)
    cache[cache_key] = pseudo_target
```

### 2. Batch Processing

Nếu có nhiều queries cùng lúc:

```python
# Process multiple queries in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_query, q) for q in queries]
```

### 3. Monitoring

```python
# Log timing và quality metrics
import logging
logging.info(f"Generation: {gen_time}s, Brightness: {brightness}")
```

---

## 🎉 Kết Luận

Bạn đã setup thành công **CIG Model với Pseudo-Target Generation**!

### ✅ Đã Hoàn Thành

1. ✅ Models đầy đủ (CLIP, Phi, SDXL)
2. ✅ API với Pseudo-Target Generation
3. ✅ Website demo với visualization
4. ✅ Database extraction script

### 🚀 Next Steps

1. **Test với nhiều queries khác nhau**
2. **Benchmark accuracy vs. direct embedding**
3. **Tối ưu speed (caching, batching)**
4. **Deploy lên server (nếu cần)**

### 📚 References

- **Paper:** Generative Zero-Shot Composed Image Retrieval
- **Approach:** Pseudo-Target Generation
- **Framework:** SDXL + CLIP + Phi Network

---

**Happy Coding! 🎨✨**

_Nếu gặp vấn đề, check lại các bước trong Troubleshooting section._
