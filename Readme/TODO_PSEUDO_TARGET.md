# ✅ TODO List - CIG Pseudo-Target Generation Setup

## 📋 Checklist để chạy được demo

### Phase 1: Kiểm tra môi trường ✅ (ĐÃ XONG)

- [x] Python 3.10 virtual environment
- [x] Dependencies đã cài (requirements.txt)
- [x] Models đã có:
  - [x] `models/phi_best.pt`
  - [x] `models/phi_best_giga.pt`
  - [x] `models/checkpoint-20000-SDXL/checkpoint-20000/unet/`

### Phase 2: Chuẩn bị Database (CẦN LÀM)

#### Option A: Nếu CHƯA có database embeddings

```powershell
# Step 1: Download images (nếu chưa có)
python download_images.py --dataset fashioniq --categories dress shirt toptee
# Time: 1-2 giờ, Size: ~50GB

# Step 2: Extract database embeddings
python extract_database_features.py --dataset fashioniq --categories dress shirt toptee --device cuda
# Time: 15-30 phút (GPU), Output: database_embeddings/fashioniq_database.pt

# Step 3: (Optional) Xóa downloaded images để tiết kiệm space
# Remove-Item -Recurse -Force datasets\FashionIQ\images\downloaded\
```

#### Option B: Nếu ĐÃ có database embeddings

```powershell
# Kiểm tra file có tồn tại
ls database_embeddings\fashioniq_database.pt
# Nếu có → Skip Phase 2
```

### Phase 3: Test và chạy demo

```powershell
# 1. Test workflow
python test_pseudo_target_workflow.py
# Expected: All tests pass

# 2. Start API server
python api_pseudo_target.py
# Keep terminal open!

# 3. Mở website
start demo_website_pseudo_target.html

# 4. Test trên website:
#    - Upload ảnh
#    - Nhập query
#    - Click "Generate & Search"
#    - Xem pseudo-target image
#    - Xem retrieval results
```

---

## 🎯 Mục tiêu cuối cùng

- [ ] API server chạy được (`python api_pseudo_target.py`)
- [ ] Website connect được API
- [ ] Upload ảnh + query → Generate pseudo-target (10-30s)
- [ ] Hiển thị pseudo-target image
- [ ] Hiển thị top-20 retrieval results
- [ ] Timing info hiển thị đúng

---

## ⚡ Quick Commands

### Activate environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Run full pipeline

```powershell
# Terminal 1: API Server
python api_pseudo_target.py

# Terminal 2: Test (optional)
curl http://localhost:5000/health

# Browser: Open demo
start demo_website_pseudo_target.html
```

---

## 🐛 Common Issues & Quick Fixes

### Issue: "Database not loaded"

```powershell
# Fix: Extract database
python extract_database_features.py
```

### Issue: "SDXL pipeline not loaded"

```powershell
# Fix: Check models exist
ls models\checkpoint-20000-SDXL\checkpoint-20000\unet\
```

### Issue: "CUDA out of memory"

```powershell
# Fix 1: Use smaller images (edit api_pseudo_target.py line ~280)
height = 384  # instead of 512

# Fix 2: Use CPU (slow)
$env:CUDA_VISIBLE_DEVICES="-1"
python api_pseudo_target.py
```

### Issue: "Cannot connect to API"

```powershell
# Fix: Check API is running
curl http://localhost:5000/health
```

---

## 📊 Files Created/Modified

### New Files ✨

- [x] `api_pseudo_target.py` - Backend với Pseudo-Target Generation
- [x] `demo_website_pseudo_target.html` - Frontend với pseudo-target display
- [x] `test_pseudo_target_workflow.py` - Test script
- [x] `Readme/PSEUDO_TARGET_SETUP.md` - Setup guide chi tiết
- [x] `Readme/APPROACH_COMPARISON.md` - So sánh 2 approaches
- [x] `Readme/QUICK_START_PSEUDO_TARGET.md` - Quick start guide

### Modified Files 📝

- [x] `config.py` - Updated SDXL checkpoint path

### Existing Files (No changes needed) ✅

- `api.py` - Old direct embedding approach
- `demo_website.html` - Old direct embedding frontend
- `extract_database_features.py` - Already exists
- `download_images.py` - Already exists
- Other files...

---

## 🎓 For Your Assignment

### What You Need:

1. **Understanding:**

   - [x] Read `APPROACH_COMPARISON.md`
   - [ ] Understand Pseudo-Target Generation workflow
   - [ ] Know why it's better than Direct Embedding

2. **Implementation:**

   - [ ] Setup và chạy được demo
   - [ ] Test với nhiều queries
   - [ ] Screenshot results

3. **Evaluation:**

   - [ ] Measure Recall@K (optional)
   - [ ] Compare với baseline
   - [ ] Analyze pseudo-target quality

4. **Report:**
   - [ ] Explain architecture
   - [ ] Show results
   - [ ] Discuss findings

---

## 📖 Documentation to Read

**Priority 1 (MUST READ):**

1. `QUICK_START_PSEUDO_TARGET.md` - Hướng dẫn setup nhanh
2. `APPROACH_COMPARISON.md` - Hiểu 2 approaches

**Priority 2 (Should read):** 3. `PSEUDO_TARGET_SETUP.md` - Chi tiết setup 4. `IMAGE_INPUT_GUIDE.md` - Guidelines về input images

**Priority 3 (Reference):** 5. `README_NEW.md` - Project overview 6. `WEBSITE_GUIDE.md` - Development guide

---

## 🚀 Estimated Time

### First time setup:

- Download images: 1-2 giờ (one-time)
- Extract database: 15-30 phút (one-time)
- Test workflow: 5 phút
- **Total: ~2-3 giờ**

### After setup (mỗi lần chạy):

- Start API: 1 phút
- Open website: immediate
- Each query: 10-30 giây
- **Total: ~1 phút để ready**

---

## 🎉 Success Criteria

### ✅ Setup successful khi:

- [ ] `python test_pseudo_target_workflow.py` passes all tests
- [ ] API health check returns "Pseudo-Target Generation"
- [ ] Website loads và connect được API
- [ ] Can generate pseudo-target images
- [ ] Search results display correctly

### ✅ Ready for demo khi:

- [ ] Tested với ít nhất 5 queries khác nhau
- [ ] Pseudo-target images có quality tốt
- [ ] Results có ý nghĩa (match với query)
- [ ] Có screenshots cho report

---

## 💬 Next Steps

1. **Nếu chưa có database:**

   ```powershell
   python download_images.py --dataset fashioniq --categories dress shirt toptee
   python extract_database_features.py
   ```

2. **Nếu đã có database:**

   ```powershell
   python test_pseudo_target_workflow.py
   python api_pseudo_target.py
   start demo_website_pseudo_target.html
   ```

3. **Test demo:**

   - Upload ảnh
   - Try example queries
   - Check pseudo-target quality
   - Verify retrieval results

4. **For assignment:**
   - Document workflow
   - Take screenshots
   - Measure metrics
   - Write report

---

**Status: Ready to start! 🎯**

_Last checkpoint: Environment setup completed ✅_
_Next: Extract database embeddings (if needed)_
