# 🖼️ Hướng Dẫn Sử Dụng Ảnh Input

## ✅ TL;DR - Tóm tắt nhanh

- **CÓ THỂ** dùng bất kỳ ảnh nào (ảnh của bạn, từ Google, từ điện thoại)
- **KHÔNG BẮT BUỘC** phải dùng ảnh trong bộ test
- **KẾT QUẢ** luôn tìm trong database FashionIQ (dress, shirt, toptee)
- **TỐT NHẤT** là ảnh quần áo (vì model train trên fashion dataset)

---

## 🎯 Các Trường Hợp Sử Dụng

### 1️⃣ Website Demo (Dùng ảnh BẤT KỲ)

**Input ảnh từ đâu cũng được:**

```
✅ Ảnh từ Google Images
✅ Ảnh chụp quần áo của bạn
✅ Ảnh từ website thời trang
✅ Ảnh screenshot từ video
✅ Ảnh từ điện thoại
```

**Ví dụ thực tế:**

```
1. Bạn có ảnh áo sơ mi xanh dương (từ đâu cũng được)
2. Upload lên website
3. Nhập query: "change to red color and add floral pattern"
4. Model sẽ:
   - Extract features từ ảnh áo xanh của bạn
   - Kết hợp với text query
   - Tìm trong 77,683 ảnh FashionIQ
   - Trả về top 10 áo đỏ có hoa giống nhất
```

**Kết quả:**

- Trả về ảnh từ FashionIQ database (có URL sẵn)
- Sorted theo similarity score (cao nhất = giống nhất)
- Không phụ thuộc vào ảnh input có trong database hay không

---

### 2️⃣ Evaluation/Testing (Dùng ảnh TRONG test set)

**Khi nào cần:**

```
❌ Không cần cho website demo
✅ Cần khi muốn tính Recall@K chính xác
✅ Cần khi so sánh với paper gốc
✅ Cần khi chạy test_CIG.py hoặc extract_lincir_feat.py
```

**Ví dụ:**

```python
# Test set có ground truth
reference_image: "dress_001.jpg"
query: "is darker blue and has longer sleeves"
target_image: "dress_045.jpg"  # Ground truth

# Đánh giá model có tìm đúng dress_045 trong top K không
```

---

## 🔍 Cách Model Hoạt Động

### Kiến trúc:

```
[Ảnh BẤT KỲ]
    ↓
[CLIP Vision Encoder] → 768-dim features
    ↓
[Phi Network] ← [Text Query: "more red"]
    ↓
[Combined Features] → 768-dim
    ↓
[Cosine Similarity] với database (77,683 ảnh)
    ↓
[Top K Results] từ FashionIQ
```

### Không cần ảnh input trong database vì:

- Model **không** search exact match
- Model search **semantic similarity** (tương tự về concept)
- CLIP đã được pre-train trên 400M ảnh → generalize tốt

---

## 🎨 Gợi Ý Ảnh Input Tốt

### ✅ TỐT (Fashion items):

```
✓ Áo sơ mi, áo thun, áo khoác
✓ Váy, đầm
✓ Quần jeans, quần âu
✓ Ảnh rõ nét, sáng, background đơn giản
✓ Góc chụp thẳng, full item
```

### ⚠️ TRUNG BÌNH:

```
△ Ảnh có nhiều items (model có thể confused)
△ Ảnh tối, mờ, góc nghiêng
△ Phụ kiện (túi, giày) - database chủ yếu là clothes
```

### ❌ KÉM:

```
✗ Động vật, phong cảnh, đồ ăn
✗ Ảnh trừu tượng
✗ Ảnh không liên quan fashion
→ Model vẫn chạy nhưng kết quả không có ý nghĩa
```

---

## 📊 So Sánh Kết Quả

### Scenario A: Ảnh TRONG database

```python
Input: "dress_001.jpg" từ FashionIQ test set
Query: "is darker"
Output:
  - Top 1: dress_045.jpg (score: 0.92)  # Có thể là ground truth
  - Top 2: dress_123.jpg (score: 0.89)
  ...

# Có thể tính Recall@10 chính xác
```

### Scenario B: Ảnh NGOÀI database (Ví dụ từ Google)

```python
Input: "my_shirt.jpg" (không có trong FashionIQ)
Query: "change to red color"
Output:
  - Top 1: shirt_456.jpg (score: 0.87)  # Áo đỏ giống nhất
  - Top 2: shirt_789.jpg (score: 0.85)
  ...

# Không có ground truth để so sánh
# Nhưng demo được khả năng tổng quát của model!
```

---

## 🚀 Quick Start - Thử Ngay

### 1. Tải ảnh test từ internet:

```bash
# Ví dụ: Tải ảnh áo sơ mi
# Hoặc chụp ảnh quần áo của bạn
```

### 2. Mở website:

```bash
# Đảm bảo API đang chạy
python api.py

# Mở demo_website.html hoặc test_website.html
```

### 3. Test với queries:

```
Input: Ảnh áo xanh
Query 1: "change to red color" → Tìm áo đỏ
Query 2: "add stripes pattern" → Tìm áo có sọc
Query 3: "make it darker and longer sleeves" → Áo tối màu, tay dài
```

### 4. Xem kết quả:

- Top 10 ảnh từ FashionIQ
- Mỗi ảnh có similarity score
- Click vào ảnh để xem to

---

## 💡 Tips & Tricks

### Để có kết quả TỐT:

1. **Ảnh input rõ ràng**: Sáng, rõ nét, full item
2. **Query cụ thể**: "change to red" > "make it different"
3. **Fashion domain**: Ảnh quần áo > ảnh random
4. **Đơn giản**: 1 item per image > nhiều items

### Nếu kết quả không như mong đợi:

```
❓ Ảnh input không phải fashion item?
→ Model train trên fashion, kết quả sẽ kém

❓ Query quá abstract ("more beautiful")?
→ Dùng query cụ thể về màu sắc, pattern, style

❓ Database chưa extract?
→ Chạy: python extract_database_features.py

❓ Model chưa download?
→ Chạy lần đầu sẽ auto download từ HuggingFace
```

---

## 📈 Performance Notes

### Database size:

- **FashionIQ**: 77,683 ảnh (dress, shirt, toptee)
- Mỗi ảnh có 768-dim embedding
- Search time: ~50-100ms trên GPU, ~500ms trên CPU

### Model capabilities:

- **Trained**: Trên FashionIQ dataset (fashion items)
- **Generalizes**: Tốt với ảnh fashion ngoài dataset
- **Limitations**: Kém với non-fashion images

---

## 🎓 Kết Luận

### Cho Website Demo:

✅ **Dùng ảnh BẤT KỲ** - Không cần trong test set
✅ Model đã generalize tốt
✅ Kết quả luôn từ FashionIQ database

### Cho Research/Evaluation:

✅ **Dùng ảnh TRONG test set** để tính metrics chính xác
✅ So sánh với ground truth
✅ Report Recall@K như trong paper

### Best Practice:

```
Website Demo: Ảnh từ đâu cũng OK (miễn là fashion)
            ↓
         API sẽ xử lý
            ↓
    Kết quả từ database (77K ảnh)
```

**Enjoy coding! 🚀**
