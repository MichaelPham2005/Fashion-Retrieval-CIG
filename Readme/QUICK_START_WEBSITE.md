# 🚀 Quick Start Guide - Website Demo

## 📋 Tóm Tắt Nhanh

### Bạn CẦN test gì?

1. ✅ **Test models hoạt động** → `test_inference_simple.py`
2. ✅ **Extract database features** → `extract_database_features.py`
3. ✅ **Run website demo** → `api.py` + `demo_website.html`

### Bạn KHÔNG CẦN:

- ❌ `extract_lincir_feat.py` (chỉ dùng cho paper evaluation)
- ❌ `test_CIG.py` (chỉ dùng cho paper - generate pseudo images)
- ❌ Training code (model đã pretrained sẵn)

---

## 🎯 Workflow Cho Website (3 Bước)

```
Step 1: Test Setup
    ↓
Step 2: Extract Database
    ↓
Step 3: Run Website
```

---

## Step 0: Cài Đặt

```bash
# Activate venv
.venv\Scripts\activate

# Install thêm Flask
pip install flask flask-cors

# Hoặc reinstall toàn bộ
pip install -r requirements.txt
```

---

## Step 1: Test Setup ⚡

### Kiểm tra models có chạy được không

```bash
python test_inference_simple.py
```

**Kết quả mong đợi:**

```
======================================================================
Simple Inference Test
======================================================================
Device: cuda

📥 Loading models...
✅ Models loaded successfully!

🖼️  Creating test image...

🔍 Extracting image features...
✅ Image features shape: torch.Size([1, 768])

🧠 Running Phi network...
✅ Pseudo tokens shape: torch.Size([1, 768])

📝 Encoding text query...
✅ Text embedding shape: torch.Size([1, 768])

======================================================================
✅ INFERENCE TEST PASSED!
======================================================================
```

**Nếu lỗi "Model not found":**

1. Download models từ [Google Drive](https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA)
2. Đặt vào thư mục `models/`:
   - `models/phi_best.pt`
   - `models/phi_best_giga.pt`

---

## Step 2: Extract Database Features 💾

### 2.1. Download Images (Nếu chưa có)

```bash
# Download 1 category để test (nhẹ hơn)
python download_images.py --dataset fashioniq --categories dress
```

**Thời gian:** ~10-15 phút
**Dung lượng:** ~15GB cho dress

### 2.2. Extract Features

```bash
# Extract features cho dress category
python extract_database_features.py \
    --dataset fashioniq \
    --categories dress \
    --output_path ./database_embeddings/fashioniq_database.pt
```

**Output:**

```
======================================================================
Database Feature Extraction
======================================================================

📥 Loading CLIP model...
✅ CLIP model loaded

📂 Processing dress...
Extracting dress: 100%|████████| 25000/25000 [10:25<00:00, 40.0it/s]
  ✅ dress: Processed 24850, Errors 150

💾 Saving database...

======================================================================
✅ Database feature extraction complete!
Total images: 24850
Output file: ./database_embeddings/fashioniq_database.pt
File size: 456.23 MB

💡 You can now DELETE downloaded images to save space:
   rm -rf ./datasets/FashionIQ/images/downloaded/
======================================================================
```

### 2.3. Xóa Downloaded Images (Optional)

```powershell
# Windows PowerShell
Remove-Item -Recurse -Force datasets\FashionIQ\images\downloaded\
```

**Tiết kiệm:** 15GB disk space!

---

## Step 3: Run Website 🌐

### 3.1. Start Backend API

```bash
python api.py
```

**Output:**

```
🚀 Starting server and loading models...
Device: cuda
📥 Loading CLIP models...
📥 Loading Phi model...
📥 Loading database embeddings...
✅ Loaded 24850 database embeddings
✅ All models loaded successfully!

======================================================================
🌐 Starting Flask API server...
======================================================================
Endpoints:
  - GET  /health  : Health check
  - POST /search  : Composed image search
  - GET  /stats   : Database statistics

Server will run on: http://localhost:5000
======================================================================

 * Running on http://0.0.0.0:5000
```

**Để server chạy (không tắt terminal này)**

### 3.2. Mở Website

**Option 1: Mở trực tiếp file HTML**

```
Mở file: demo_website.html trong browser
```

**Option 2: Dùng live server (VSCode)**

```
Right-click demo_website.html → Open with Live Server
```

### 3.3. Test Website

1. **Upload ảnh reference** (áo dress)
2. **Nhập query:** "more red color"
3. **Click "Search"**
4. **Xem kết quả:** 20 ảnh áo đỏ tương tự

**Response time:** ~0.3-0.5 giây

---

## 🎬 Demo Scenarios

### Scenario 1: Tìm áo đỏ hơn

```
Reference: Áo trắng
Query: "more red color"
Result: 20 áo đỏ
```

### Scenario 2: Tìm áo không tay

```
Reference: Áo dài tay
Query: "without sleeves"
Result: 20 áo không tay
```

### Scenario 3: Tìm áo dài hơn và tối hơn

```
Reference: Áo ngắn sáng màu
Query: "longer and darker"
Result: 20 áo dài màu tối
```

---

## 🔧 Troubleshooting

### Lỗi: "Database not loaded"

**Nguyên nhân:** Chưa chạy Step 2
**Giải pháp:**

```bash
python extract_database_features.py --categories dress
```

### Lỗi: "API server not responding"

**Nguyên nhân:** Backend chưa start
**Giải pháp:**

```bash
python api.py
```

### Lỗi: "CUDA out of memory"

**Giải pháp:**

```bash
# Sửa trong api.py, line ~25:
device = 'cpu'  # Thay vì 'cuda'
```

### Lỗi: "Image not found" trong results

**Nguyên nhân:** URL cũ/broken
**Giải pháp:** Normal, một số URLs có thể bị broken

### Website không load ảnh

**Nguyên nhân:** CORS hoặc URLs broken
**Giải pháp:** Check console (F12), URLs có thể cần proxy

---

## 📊 Performance Expectations

### Hardware Requirements:

- **GPU:** RTX 3060+ (12GB VRAM recommended)
- **RAM:** 16GB+
- **Disk:** 20GB (giảm xuống 1GB sau khi xóa images)

### Processing Time:

- **Test inference:** ~5 seconds
- **Extract database:** ~10-20 minutes/category
- **Website search:** ~0.3-0.5 seconds/query

### Database Size:

- **1 category (dress):** ~25k images → 450MB embeddings
- **3 categories (all):** ~50k images → 900MB embeddings

---

## 💡 Tips

### Tip 1: Test với 1 category trước

```bash
# Chỉ download dress (nhẹ nhất)
python download_images.py --categories dress
python extract_database_features.py --categories dress
```

### Tip 2: Kiểm tra API health

```bash
# Trong browser hoặc terminal
curl http://localhost:5000/health
```

**Response:**

```json
{
  "status": "healthy",
  "device": "cuda",
  "database_size": 24850
}
```

### Tip 3: Test API trực tiếp

```bash
curl -X POST http://localhost:5000/search \
  -F "image=@test_image.jpg" \
  -F "query=more red color"
```

---

## 📁 File Structure Sau Khi Setup

```
ComposedImageGen/
├── models/
│   ├── phi_best.pt              ✅ (downloaded)
│   └── phi_best_giga.pt         ✅ (downloaded)
│
├── database_embeddings/
│   └── fashioniq_database.pt    ✅ (created by extract)
│
├── datasets/FashionIQ/
│   ├── images/
│   │   ├── dress.json           ✅ (URLs)
│   │   └── downloaded/          ❌ (deleted to save space)
│   └── captions/                ✅ (có sẵn)
│
├── api.py                       ✅ (backend)
├── demo_website.html            ✅ (frontend)
├── test_inference_simple.py     ✅ (test script)
└── extract_database_features.py ✅ (extract script)
```

---

## ✅ Checklist

Trước khi chạy website:

- [ ] Models đã download (phi_best.pt, phi_best_giga.pt)
- [ ] Test inference passed (`test_inference_simple.py`)
- [ ] Images đã download (ít nhất 1 category)
- [ ] Database features đã extract (`.pt` file)
- [ ] Flask đã cài (`pip install flask flask-cors`)
- [ ] Backend đang chạy (`python api.py`)
- [ ] Website đã mở (`demo_website.html`)

---

## 🎯 Kết Luận

### Workflow Đơn Giản:

```
Test → Extract → Run API → Mở Website → Done!
```

### Time Investment:

- **Setup lần đầu:** 30-40 phút
- **Sau đó:** Chỉ cần start API (30 giây)

### Có thể demo ngay:

1. ✅ Upload ảnh bất kỳ
2. ✅ Nhập text modification
3. ✅ Nhận 20 results trong 0.5s
4. ✅ Show cho người khác xem

**Không cần:**

- ❌ Train model
- ❌ Generate pseudo images (test_CIG.py)
- ❌ Full dataset (1 category đủ)
- ❌ Giữ downloaded images

---

Chúc bạn thành công! 🎉
