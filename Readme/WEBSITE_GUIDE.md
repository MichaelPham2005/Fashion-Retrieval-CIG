# 🌐 Hướng Dẫn Setup cho Website GUI Demo

## 🎯 Mục Đích: Xây Dựng Website Composed Image Retrieval

**Chức năng website:**

- User upload ảnh reference
- User nhập text query: "Tôi muốn ảnh này nhưng..."
- System trả về list ảnh matching từ database

---

## 📋 Workflow Tối Ưu cho Website

### 🔄 Workflow Inference (Real-time)

```
User Input
  ├─ Reference Image
  └─ Text Query: "more red, sleeveless"
        ↓
   CLIP Vision Encoder (real-time)
        ↓
   Phi Network (real-time)
        ↓
   CLIP Text Encoder (real-time)
        ↓
   Composed Embedding
        ↓
   Compare với Pre-computed Database Embeddings
        ↓
   Return Top-K Similar Images (URLs)
```

### 💾 Database Preparation (Offline - 1 lần duy nhất)

```
All Database Images (URLs)
        ↓
   Download images (tạm thời)
        ↓
   Extract Visual Features → Save embeddings
        ↓
   XÓA downloaded images ✅
        ↓
   Giữ lại: embeddings + URLs
```

---

## 🚀 Setup Instructions cho Website

### Phase 1: Chuẩn Bị Database (OFFLINE - Chạy 1 lần)

#### Step 1: Download Database Images (Tạm thời)

```bash
python download_images.py --dataset fashioniq --categories dress shirt toptee
```

**Lưu ý:** Ảnh này sẽ XÓA sau bước 2

#### Step 2: Extract Database Features

```bash
# Extract features cho TẤT CẢ ảnh trong database
python extract_database_features.py \
    --dataset fashioniq \
    --categories dress shirt toptee \
    --output_dir ./database_embeddings
```

**Output:**

- File: `database_embeddings/fashioniq_all.pt`
- Chứa: Dict mapping `asin → embedding`
- Size: ~500MB cho 50k images

#### Step 3: XÓA Downloaded Images

```bash
# Tiết kiệm disk space
rm -rf datasets/FashionIQ/images/downloaded/
```

**Kết quả sau Phase 1:**

```
✅ Có: database_embeddings/fashioniq_all.pt (embeddings)
✅ Có: datasets/FashionIQ/images/dress.json (URLs)
❌ Không cần: Downloaded images (đã xóa)
```

---

### Phase 2: Web Backend API (RUNTIME)

#### Backend Architecture

```python
# api.py - Flask/FastAPI backend

from flask import Flask, request, jsonify
import torch
from PIL import Image
import clip
from phi import Phi
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

app = Flask(__name__)

# Load models khi start server (1 lần)
clip_model = CLIPVisionModelWithProjection.from_pretrained(...)
phi_model = Phi(...)
database_embeddings = torch.load('database_embeddings/fashioniq_all.pt')
image_urls = load_image_urls('datasets/FashionIQ/images/dress.json')

@app.route('/search', methods=['POST'])
def composed_search():
    # Nhận input từ frontend
    ref_image = request.files['image']
    text_query = request.form['query']

    # 1. Extract features từ reference image (real-time)
    img = Image.open(ref_image)
    img_features = clip_model(preprocess(img))

    # 2. Phi network
    pseudo_tokens = phi_model(img_features)

    # 3. Combine với text query
    text_embedding = encode_text_with_pseudo_tokens(text_query, pseudo_tokens)

    # 4. Search trong database
    similarities = compute_similarity(text_embedding, database_embeddings)
    top_k_indices = torch.topk(similarities, k=20).indices

    # 5. Return URLs (không cần ảnh thật!)
    results = [
        {
            'url': image_urls[idx],
            'score': similarities[idx].item()
        }
        for idx in top_k_indices
    ]

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
```

#### Frontend (React/Vue/HTML)

```javascript
// Frontend gọi API
async function searchImages(referenceImage, textQuery) {
  const formData = new FormData();
  formData.append("image", referenceImage);
  formData.append("query", textQuery);

  const response = await fetch("http://localhost:5000/search", {
    method: "POST",
    body: formData,
  });

  const results = await response.json();

  // Display images từ URLs
  results.forEach((result) => {
    displayImage(result.url, result.score);
  });
}
```

---

## ❓ FAQ: test_CIG.py CÓ CẦN CHO WEBSITE KHÔNG?

### Câu trả lời: **KHÔNG BẮT BUỘC**

**2 Approaches:**

### Approach 1: Direct Embedding Comparison (Đơn giản hơn)

```
User Query → Composed Embedding → Compare Database → Return URLs
```

**Ưu điểm:**

- ✅ Nhanh (không cần generate image)
- ✅ Đơn giản
- ✅ Đủ tốt cho demo

**Nhược điểm:**

- ⚠️ Accuracy có thể thấp hơn một chút

### Approach 2: Với Pseudo-Target Generation (Phức tạp hơn)

```
User Query → Composed Embedding → Generate Pseudo Image (test_CIG.py)
→ Extract Pseudo Image Features → Compare Database → Return URLs
```

**Ưu điểm:**

- ✅ Accuracy cao hơn (theo paper)
- ✅ Có thể show pseudo image cho user xem

**Nhược điểm:**

- ⚠️ Chậm hơn (phải generate image: ~3-5s)
- ⚠️ Cần GPU mạnh hơn
- ⚠️ Phức tạp hơn

**Khuyến nghị:** Bắt đầu với **Approach 1**, sau đó nâng cấp lên Approach 2 nếu cần.

---

## 📦 Files Cần Thiết cho Website

### Cần có:

```
✅ models/phi_best.pt              (Phi model)
✅ models/phi_best_giga.pt         (Phi model)
✅ database_embeddings/*.pt        (Pre-computed embeddings)
✅ datasets/FashionIQ/images/*.json (URLs mapping)
✅ api.py                          (Backend code)
✅ frontend/                       (Website code)
```

### KHÔNG cần:

```
❌ datasets/FashionIQ/images/downloaded/  (Đã xóa sau extract)
❌ outputs/embeddings/                    (Chỉ dùng cho test_CIG.py)
❌ outputs/generated_images/              (Chỉ dùng cho evaluation)
❌ models/sdxl_checkpoint/                (Chỉ cần nếu dùng Approach 2)
```

---

## 💻 Script để Tạo Database Embeddings

### Tôi sẽ tạo script mới: `extract_database_features.py`

```python
# extract_database_features.py
"""
Extract visual features cho TẤT CẢ ảnh trong database
Chỉ chạy 1 lần khi setup website
"""

import torch
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from tqdm import tqdm
import json
import os

def extract_database_features(
    dataset_path='./datasets/FashionIQ',
    categories=['dress', 'shirt', 'toptee'],
    output_path='./database_embeddings/fashioniq_all.pt'
):
    # Load CLIP model
    model = CLIPVisionModelWithProjection.from_pretrained(
        'openai/clip-vit-large-patch14'
    ).cuda()

    database = {}  # asin → embedding

    for category in categories:
        # Load URLs
        json_path = f'{dataset_path}/images/{category}.json'
        with open(json_path) as f:
            items = json.load(f)

        # Load split để biết images nào trong database
        split_path = f'{dataset_path}/image_splits/split.{category}.test.json'
        with open(split_path) as f:
            test_asins = set(json.load(f))

        # Extract features
        for item in tqdm(items, desc=f'Extracting {category}'):
            asin = item['asin']
            if asin not in test_asins:
                continue

            # Load image
            img_path = f'{dataset_path}/images/downloaded/{category}/{asin}.jpg'
            img = Image.open(img_path).convert('RGB')

            # Extract features
            with torch.no_grad():
                features = model(preprocess(img)).image_embeds

            database[asin] = features.cpu()

    # Save
    torch.save(database, output_path)
    print(f'Saved {len(database)} embeddings to {output_path}')

if __name__ == '__main__':
    extract_database_features()
```

---

## 🎬 Demo Workflow

### User Experience:

```
1. User mở website
2. Upload ảnh áo: [Áo trắng]
3. Nhập query: "more red color, without sleeves"
4. Click "Search"
   ↓
5. Backend:
   - Extract features từ áo trắng (0.1s)
   - Phi network (0.05s)
   - Combine với "more red..." (0.05s)
   - Search database 50k images (0.1s)
   ↓
6. Return top 20 URLs (total: ~0.3s)
7. Frontend display 20 ảnh áo đỏ không tay
```

**Response time:** < 0.5s (rất nhanh!)

---

## 📊 So Sánh Storage

### Nếu GIỮ downloaded images:

```
Downloaded images:     50 GB
Embeddings:           0.5 GB
Total:               50.5 GB
```

### Nếu XÓA downloaded images:

```
Embeddings only:     0.5 GB
URLs (JSON):         0.01 GB
Total:               0.51 GB (tiết kiệm 99%!)
```

---

## 🔧 Next Steps cho Bạn

### 1. Test Models (Hiện tại)

```bash
# Test xem models có chạy được không
python test_inference_simple.py
```

### 2. Extract Database (1 lần)

```bash
# Tạo database embeddings
python extract_database_features.py
```

### 3. Xóa Images (Sau khi extract xong)

```bash
rm -rf datasets/FashionIQ/images/downloaded/
```

### 4. Build API

```bash
# Tạo Flask/FastAPI backend
python api.py
```

### 5. Build Frontend

```bash
# React/Vue website
npm run dev
```

---

## 💡 Tóm Tắt

### Bạn CẦN:

1. ✅ Extract database features (1 lần) → `.pt` files
2. ✅ URLs mapping (có sẵn)
3. ✅ Backend API (real-time inference)
4. ✅ Frontend website

### Bạn KHÔNG CẦN:

1. ❌ Downloaded images (xóa sau extract)
2. ❌ test_CIG.py (optional, dùng cho paper evaluation)
3. ❌ SDXL models (nếu dùng Approach 1)
4. ❌ Training code

### Workflow Đơn Giản:

```
Download → Extract → XÓA images → Build API → Done!
```

---

Bạn muốn tôi tạo script `extract_database_features.py` và `api.py` mẫu không?
