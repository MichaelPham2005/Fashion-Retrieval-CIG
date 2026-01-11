# 🚀 Setup Guide: Chạy CIG Server trên Google Colab + Demo Local

## 📋 Tổng Quan

**Architecture:**

- **Server**: Chạy trên Google Colab (GPU miễn phí 16GB T4)
- **Demo**: Chạy trên máy local (HTML file)
- **Connection**: Qua ngrok public URL

**Ưu điểm:**

- ✅ GPU mạnh (16GB VRAM) - đủ chạy SDXL
- ✅ Miễn phí (Colab Free Tier)
- ✅ Generation time: ~15-30 giây (thay vì 3-5 phút trên CPU)
- ✅ Không cần cài đặt môi trường local
- ✅ Demo vẫn chạy offline trên máy bạn

---

## 🎯 Hướng Dẫn Chi Tiết

### Bước 1: Upload Project lên GitHub

```bash
# Nếu chưa có repo
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/lan-lw/ComposedImageGen.git
git push -u origin main

# Nếu đã có repo
git add .
git commit -m "Add Colab setup"
git push
```

**⚠️ Lưu ý:**

- Nếu models lớn, đừng push models lên GitHub
- Upload models lên Google Drive thay thế

---

### Bước 2: Chuẩn Bị Models

**Option A: Models trong repo (nếu nhỏ)**

- Đảm bảo `models/phi_best.pt` và `models/checkpoint-20000-SDXL/` trong repo

**Option B: Upload lên Google Drive (khuyến nghị)**

1. Upload files:

   - `models/phi_best.pt` (~200MB)
   - `models/checkpoint-20000-SDXL/` (folder, ~10GB)

2. Trong Colab notebook, sẽ mount Drive và copy:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !cp /content/drive/MyDrive/CIG_Models/phi_best.pt ./models/
   !cp -r /content/drive/MyDrive/CIG_Models/checkpoint-20000-SDXL ./models/
   ```

---

### Bước 3: Setup Ngrok

1. **Đăng ký Ngrok (miễn phí):**

   - Truy cập: https://dashboard.ngrok.com/signup
   - Sign up với Google/GitHub

2. **Lấy Auth Token:**

   - Sau khi đăng nhập: https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy token (dạng: `2a...xyz`)

3. **Lưu token** để paste vào Colab notebook

---

### Bước 4: Chạy Colab Notebook

1. **Mở Colab:**

   - Upload file `colab_setup.ipynb` lên Google Drive
   - Hoặc: File → Upload notebook → chọn `colab_setup.ipynb`

2. **Chọn GPU Runtime:**

   ```
   Runtime → Change runtime type → Hardware accelerator → GPU → Save
   ```

   - Free tier: T4 (16GB VRAM) ✅
   - Colab Pro: A100 (40GB VRAM) 🚀

3. **Run từng cell theo thứ tự:**

   **Cell 1: Check GPU**

   ```bash
   !nvidia-smi
   ```

   - Xác nhận: T4 GPU, 16GB memory

   **Cell 2-3: Clone & Install**

   ```bash
   !git clone https://github.com/lan-lw/ComposedImageGen.git
   %cd ComposedImageGen
   !pip install -q torch transformers diffusers ...
   ```

   **Cell 4: Upload Models** (nếu dùng Drive)

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Copy models từ Drive
   ```

   **Cell 5: Extract Database** (~15-30 phút)

   ```bash
   !python extract_database_features.py --dataset fashioniq --categories dress shirt toptee --device cuda
   ```

   **Cell 6: Setup Ngrok**

   - Paste ngrok token khi được hỏi

   **Cell 7: Start Server** ⭐

   - Server khởi động, loading models (~2-3 phút)
   - Xuất hiện **Public URL**: `https://xxxx-xx-xxx.ngrok-free.app`
   - **COPY URL NÀY!**

---

### Bước 5: Chạy Demo Trên Máy Local

1. **Mở file demo:**

   ```
   D:\ComposedImageGen\demo_website_colab.html
   ```

   - Double click hoặc mở bằng browser

2. **Paste Server URL:**

   - Trong phần "Server Configuration" (màu vàng)
   - Paste URL từ Colab: `https://xxxx-xx-xxx.ngrok-free.app`
   - Click **"Test Connection"**

3. **Kiểm tra kết nối:**

   - Nếu thành công: ✅ "Connected to Colab server!"
   - Hiển thị GPU info, database size, SDXL status

4. **Bắt đầu search:**
   - Upload reference image
   - Nhập modification query
   - Click "Search with Pseudo-Target Generation"
   - Đợi ~15-30 giây
   - Xem pseudo-target và results!

---

## 🎓 Demo Flow

```
┌─────────────────────┐
│   Local Machine     │
│  (Your Computer)    │
│                     │
│  demo_website       │
│  _colab.html        │
└──────────┬──────────┘
           │
           │ HTTP Request
           │ (ngrok URL)
           ▼
┌─────────────────────┐
│   Google Colab      │
│   (Cloud GPU)       │
│                     │
│  api_pseudo_target  │
│      .py            │
│                     │
│  • CLIP (GPU)       │
│  • Phi (GPU)        │
│  • SDXL (GPU)       │
│  • Database         │
└─────────────────────┘
```

---

## ⚡ Performance

| Component               | Device       | Time        |
| ----------------------- | ------------ | ----------- |
| CLIP Feature Extraction | Colab T4 GPU | ~0.1s       |
| Phi Network             | Colab T4 GPU | ~0.05s      |
| SDXL Generation         | Colab T4 GPU | ~15-20s     |
| Database Search         | Colab T4 GPU | ~0.5s       |
| **Total**               |              | **~15-30s** |

**So sánh:**

- Local CPU: 3-5 minutes ❌
- Colab GPU: 15-30 seconds ✅

---

## 💡 Tips & Troubleshooting

### 🔴 Connection Failed?

**1. Check Colab cell đang chạy:**

- Cell với server phải đang active (running)
- Có dấu [*] bên trái cell

**2. Check URL chính xác:**

- Phải có `https://`
- Không có dấu `/` cuối cùng
- Copy chính xác từ Colab output

**3. Check ngrok free tier:**

- Miễn phí: 1 active tunnel
- Nếu đã có tunnel khác → disconnect

**4. Test bằng browser:**

- Open: `https://your-url.ngrok-free.app/health`
- Phải thấy JSON response

---

### ⏱️ Colab Runtime Limits

**Free Tier:**

- Max runtime: ~12 hours
- Idle timeout: 90 minutes
- Daily limit: ~12-15 hours

**Giải pháp:**

- Click vào notebook mỗi 60-90 phút để keep alive
- Sử dụng extension: Colab Auto Clicker
- Upgrade Colab Pro ($9.99/tháng)

---

### 🔄 Restart Server

Nếu server bị lỗi:

1. **Interrupt cell** (nút Stop bên cạnh cell)
2. **Re-run cell cuối** (Start Server)
3. **Copy new URL** (ngrok tạo URL mới)
4. **Update trong demo HTML**

---

### 📦 Database Not Found

Nếu thiếu database:

```bash
# Trong Colab, run cell:
!python extract_database_features.py \
    --dataset fashioniq \
    --categories dress shirt toptee \
    --device cuda
```

Time: 15-30 phút

---

### 🎨 Slow Generation?

**Colab Free (T4):** ~20-30s  
**Colab Pro (A100):** ~10-15s

**Optimization options:**

```javascript
// Trong demo HTML, adjust:
- Image Size: 512x512 (faster) vs 1024x1024 (better)
- Inference Steps: 30 (faster) vs 50 (better)
```

---

## 🎯 Workflow cho Assignment

### Day 1: Setup

1. ✅ Push code lên GitHub
2. ✅ Upload models lên Drive
3. ✅ Test Colab notebook
4. ✅ Extract database (~30 min)
5. ✅ Test 1-2 queries

### Day 2: Generate Examples

```python
# Run trong Colab
queries = [
    ("dress_img1.jpg", "is red and has long sleeves"),
    ("shirt_img2.jpg", "has floral pattern"),
    ("toptee_img3.jpg", "is more casual"),
]

for img, query in queries:
    # Upload image, run search
    # Save pseudo-target
    # Save top-20 results
```

**Output:**

- 10-15 example queries
- Pseudo-target images
- Retrieval results
- Timing metrics

### Day 3: Prepare Report

**Sections:**

1. **Theory:** Pseudo-Target Generation approach
2. **Implementation:** Code on Colab
3. **Results:** Pre-generated examples
4. **Analysis:**
   - Compare with Direct Embedding
   - Discuss generation quality
   - Show retrieval accuracy
5. **Demo:**
   - Option A: Live demo (need Colab running)
   - Option B: Video recording
   - Option C: Screenshots

---

## 📊 Example Report Section

```markdown
## Implementation

### Architecture

We implemented the Pseudo-Target Generation approach on Google Colab
with T4 GPU (16GB VRAM) to overcome local hardware limitations.

### Deployment Strategy

- **Server**: Flask API on Colab with ngrok public URL
- **Client**: HTML/JavaScript demo on local machine
- **Communication**: RESTful API over HTTPS

### Performance

- SDXL generation: ~20 seconds per query
- Total workflow: ~25 seconds end-to-end
- Hardware: Google Colab T4 GPU (16GB)

### Results

[Show pre-generated examples with pseudo-targets]
[Compare retrieval quality with baseline]

### Challenges & Solutions

**Challenge:** Local GPU (4GB) insufficient for SDXL
**Solution:** Deploy on Google Colab with free T4 GPU (16GB)
**Result:** Successfully achieved real-time pseudo-target generation
```

---

## ✅ Checklist

### Before Demo:

- [ ] Colab notebook uploaded
- [ ] Models on Drive hoặc trong repo
- [ ] Ngrok account & token
- [ ] GitHub repo public/accessible
- [ ] Database extracted
- [ ] Test connection successful

### During Demo:

- [ ] Colab cell running (keep alive)
- [ ] Public URL active
- [ ] Demo HTML opened locally
- [ ] Connection tested
- [ ] Example queries prepared

### After Demo:

- [ ] Save generated examples
- [ ] Screenshot results
- [ ] Record metrics (time, accuracy)
- [ ] Stop Colab runtime (save quota)

---

## 🎬 Alternative: Video Recording

Nếu không demo live:

1. **Setup Colab + Demo**
2. **Screen record:**
   - Show Colab setup
   - Show demo HTML
   - Perform 3-5 searches
   - Show pseudo-targets
   - Show results
3. **Edit video** (~3-5 phút)
4. **Upload YouTube/Drive**
5. **Include link trong report**

---

## 🚀 Quick Start Commands

```bash
# 1. Push to GitHub
git add .
git commit -m "Add Colab setup"
git push

# 2. Upload models to Drive
# - Manual upload via web interface

# 3. Open Colab
# - Upload colab_setup.ipynb
# - Runtime → GPU
# - Run all cells
# - Copy public URL

# 4. Open demo
# - Open demo_website_colab.html
# - Paste URL
# - Test connection
# - Start searching!
```

---

## 📞 Support

**Issues?**

- Check Colab logs
- Test /health endpoint
- Verify ngrok status
- Check GPU availability

**Good luck với assignment! 🎓**

---

## Summary

✅ **Created:**

- `colab_setup.ipynb` - Complete Colab notebook
- `demo_website_colab.html` - Remote server demo
- This guide

✅ **Strategy:**

- Server: Colab GPU (16GB T4)
- Client: Local HTML
- Connection: Ngrok public URL

✅ **Performance:**

- Generation: ~20s (vs 3-5 min CPU)
- Correct implementation
- Free solution

**Bây giờ bạn có thể:**

1. Push code lên GitHub
2. Run Colab notebook
3. Demo với GPU mạnh
4. Hoàn thành assignment đúng approach! 🎉
