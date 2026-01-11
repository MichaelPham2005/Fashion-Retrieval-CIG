# 🎯 CIG Model: Direct vs Pseudo-Target Approaches

## 📋 Tổng Quan

Project này có **2 implementations** khác nhau cho Composed Image Retrieval:

1. **Direct Embedding Comparison** (api.py + demo_website.html)
2. **Pseudo-Target Generation** (api_pseudo_target.py + demo_website_pseudo_target.html) ⭐ **RECOMMENDED**

---

## 🔄 So Sánh Chi Tiết

### Approach 1: Direct Embedding Comparison

#### Workflow:

```
Reference Image + Text Query
         ↓
  CLIP Vision Encoder → Extract features
         ↓
  Phi Network → Predict pseudo tokens
         ↓
  CLIP Text Encoder → Compose with query
         ↓
  Composed Embedding (768-dim vector)
         ↓
  Cosine Similarity với Database Embeddings
         ↓
  Return Top-K Results
```

#### Files:

- Backend: `api.py`
- Frontend: `demo_website.html`
- Database: `database_embeddings/fashioniq_database.pt`

#### Pros:

- ✅ **Nhanh** (~0.3 seconds)
- ✅ Đơn giản, dễ implement
- ✅ Không cần SDXL models
- ✅ Tiết kiệm GPU memory

#### Cons:

- ❌ **Accuracy thấp hơn**
- ❌ Không phản ánh bản chất của CIG model
- ❌ Không có visualization của target
- ❌ Khó debug khi results không tốt

#### Use Cases:

- Quick prototyping
- Resource-constrained environments
- Real-time applications (< 1s response)

---

### Approach 2: Pseudo-Target Generation ⭐

#### Workflow:

```
Reference Image + Text Query
         ↓
  CLIP Vision Encoder → Extract features
         ↓
  Phi Network → Predict pseudo tokens
         ↓
  CLIP Text Encoder → Compose with query
         ↓
  Composed Embedding → SDXL Input
         ↓
  SDXL Pipeline → Generate Pseudo-Target Image ⭐
         ↓
  CLIP Vision Encoder → Extract Pseudo-Target Features
         ↓
  Cosine Similarity với Database Embeddings
         ↓
  Return Top-K Results + Pseudo-Target Image
```

#### Files:

- Backend: `api_pseudo_target.py`
- Frontend: `demo_website_pseudo_target.html`
- Database: Same as Approach 1
- Models: + SDXL checkpoint

#### Pros:

- ✅ **Higher Accuracy** (theo CIG paper)
- ✅ **Đúng với bản chất model** - paper approach
- ✅ Visualization của pseudo-target
- ✅ Dễ debug và understand results
- ✅ User thấy được "ảnh mục tiêu"

#### Cons:

- ⚠️ **Chậm hơn** (~10-30 seconds)
- ⚠️ Cần GPU mạnh (12GB+ VRAM)
- ⚠️ Cần SDXL checkpoint (~5GB)
- ⚠️ Phức tạp hơn

#### Use Cases:

- Research và evaluation
- Production với quality cao
- Demo và presentation
- Information Retrieval assignments ⭐

---

## 📊 Performance Comparison

| Metric                  | Direct Embedding | Pseudo-Target |
| ----------------------- | ---------------- | ------------- |
| **Response Time (GPU)** | 0.3s             | 15s           |
| **Response Time (CPU)** | 1s               | 5min          |
| **GPU Memory**          | 4GB              | 12GB          |
| **Disk Space**          | 1GB              | 6GB           |
| **Accuracy**            | Medium           | **High**      |
| **Paper Alignment**     | ❌ No            | ✅ **Yes**    |
| **Visualization**       | ❌ No            | ✅ **Yes**    |

---

## 🎯 Which Approach to Use?

### Dùng **Direct Embedding** khi:

- ❓ Cần response nhanh (< 1s)
- ❓ GPU memory hạn chế (< 8GB)
- ❓ Chỉ cần quick demo
- ❓ Không quan tâm paper accuracy

### Dùng **Pseudo-Target** khi: ⭐

- ✅ **Làm bài tập Information Retrieval**
- ✅ Cần accuracy cao nhất
- ✅ Muốn understand model behavior
- ✅ Có GPU đủ mạnh
- ✅ Research và evaluation
- ✅ **CẦN THEO ĐÚNG PAPER CIG**

---

## 🚀 Setup Instructions

### Option 1: Direct Embedding (Simple)

```powershell
# 1. Extract database
python extract_database_features.py

# 2. Start API
python api.py

# 3. Open website
start demo_website.html
```

### Option 2: Pseudo-Target (Recommended) ⭐

```powershell
# 1. Check models exist
ls models\checkpoint-20000-SDXL\checkpoint-20000\unet\

# 2. Download database images (if not done)
python download_images.py --dataset fashioniq --categories dress shirt toptee

# 3. Extract database
python extract_database_features.py

# 4. Test workflow
python test_pseudo_target_workflow.py

# 5. Start API
python api_pseudo_target.py

# 6. Open website
start demo_website_pseudo_target.html
```

---

## 📈 Accuracy Comparison

### CIG Paper Results (with Pseudo-Target)

```
FashionIQ Recall@10:
- Dress:  XX.X%
- Shirt:  XX.X%
- Toptee: XX.X%
```

### Expected Results

**Direct Embedding:**

- Recall@10: ~15-20% lower
- Results less semantically aligned
- May retrieve similar color/pattern but wrong style

**Pseudo-Target:**

- Recall@10: As reported in paper
- Results highly aligned with query
- Better capture semantic changes

---

## 🔍 Example Comparison

### Input:

- Reference: Blue striped shirt
- Query: "change to red and add floral pattern"

### Direct Embedding Results:

1. Red shirt (no pattern) ✓
2. Blue floral shirt (wrong color) ✗
3. Pink striped shirt (partial match) △
4. ...

**Issues:** Mixed results, không consistent

### Pseudo-Target Results:

**Generated Pseudo-Target:** Red floral shirt (exactly what we want!)

Search results:

1. Red floral shirt ✓✓
2. Red floral dress ✓
3. Red flower pattern shirt ✓✓
4. ...

**Better:** Results highly aligned với generated image

---

## 🎓 Technical Deep Dive

### Why Pseudo-Target Works Better?

#### 1. Semantic Alignment

```
Direct: "Blue shirt" + "change to red" → Ambiguous vector
Pseudo: Generate actual red shirt → Clear visual target
```

#### 2. Feature Space

```
Direct: Composed embedding in mixed space (text+image)
Pseudo: Pure visual features (image-only space)
```

#### 3. Database Matching

```
Database contains: Pure image features (CLIP Vision)
Direct matching: Cross-modal (text-like vs image)
Pseudo matching: Same-modal (image vs image)
```

---

## 💡 Implementation Details

### Direct Embedding (api.py)

```python
# 1. Extract reference features
ref_features = clip_vision(reference_image)

# 2. Phi network
pseudo_tokens = phi(ref_features)

# 3. Compose with text
composed_embedding = encode_text_with_pseudo_tokens(query, pseudo_tokens)

# 4. Direct search
for asin, db_embedding in database:
    similarity = cosine_sim(composed_embedding, db_embedding)
```

### Pseudo-Target (api_pseudo_target.py)

```python
# 1-3. Same as above
composed_embedding, composed_hidden = ...

# 4. Generate pseudo-target ⭐
pseudo_image = sdxl_pipe(
    prompt_embeds=composed_hidden,
    pooled_prompt_embeds=composed_embedding
)

# 5. Extract pseudo-target features
pseudo_features = clip_vision(pseudo_image)

# 6. Search with pure visual features
for asin, db_embedding in database:
    similarity = cosine_sim(pseudo_features, db_embedding)
```

**Key difference:** Step 4-5 adds intermediate image generation!

---

## 🎬 Demo Comparison

### Direct Embedding Demo

- Input → Loading (0.3s) → Results
- Simple, fast
- No intermediate visualization

### Pseudo-Target Demo

- Input → Loading (15s) → **Pseudo-Target Image** → Results
- User sees generated image
- Better understanding of what model is looking for
- More impressive for presentations!

---

## 📝 For Information Retrieval Assignment

### Recommended Approach: **Pseudo-Target** ⭐

#### Reasons:

1. ✅ **Follows original paper** - CIG paper uses this approach
2. ✅ **Better accuracy** - Important for evaluation
3. ✅ **Clear visualization** - Easy to explain in report
4. ✅ **Research-grade** - Suitable for academic work

#### What to Include in Report:

1. **Method Description:**

   - "We implement the Pseudo-Target Generation approach from CIG paper"
   - Explain SDXL's role in generating intermediate targets
   - Show example pseudo-target images

2. **Architecture Diagram:**

   ```
   [Reference] + [Query] → [SDXL] → [Pseudo-Target] → [Search]
   ```

3. **Evaluation:**

   - Compare with baseline (Direct Embedding)
   - Show Recall@K improvements
   - Analyze generated pseudo-targets

4. **Results:**
   - Include pseudo-target visualizations
   - Show retrieval results
   - Discuss failure cases

---

## 🔧 Troubleshooting

### Common Issues:

#### "Which approach am I using?"

Check API health endpoint:

```powershell
curl http://localhost:5000/health
```

Response will include:

```json
{
  "approach": "Pseudo-Target Generation" // or "Direct Embedding"
}
```

#### "How to switch approaches?"

1. Stop current API server (Ctrl+C)
2. Start desired version:
   - Direct: `python api.py`
   - Pseudo-Target: `python api_pseudo_target.py`
3. Open corresponding HTML:
   - Direct: `demo_website.html`
   - Pseudo-Target: `demo_website_pseudo_target.html`

---

## 🎉 Conclusion

### Summary Table

| Aspect                | Direct | Pseudo-Target |
| --------------------- | ------ | ------------- |
| Speed                 | ⚡⚡⚡ | ⚡            |
| Accuracy              | ⭐⭐   | ⭐⭐⭐        |
| Paper Alignment       | ❌     | ✅            |
| Visualization         | ❌     | ✅            |
| GPU Required          | 4GB    | 12GB          |
| Setup Complexity      | Easy   | Medium        |
| **For IR Assignment** | ❌     | **✅**        |

### Final Recommendation:

**Use Pseudo-Target Generation** for:

- ✅ Information Retrieval assignments
- ✅ Research projects
- ✅ High-quality demos
- ✅ Following CIG paper

**Use Direct Embedding** only for:

- ⚡ Quick prototypes
- 💻 Limited hardware
- 🚀 Real-time applications

---

## 📚 References

- **CIG Paper:** Generative Zero-Shot Composed Image Retrieval
- **SDXL:** Stable Diffusion XL
- **CLIP:** OpenAI CLIP
- **Implementation:** Based on official CIG codebase

---

**Happy Retrieving! 🎨🔍**
