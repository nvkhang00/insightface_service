# Kiến trúc tổng thể project `face.db`

Đây là một **API nhận diện khuôn mặt học sinh** (Student Face Recognition) viết bằng Python, đóng gói bằng Docker. Mục đích: cho phép đăng ký ảnh khuôn mặt cho từng học sinh, sau đó dùng ảnh mới để xác minh (verify) xem có phải học sinh đó không, hoặc tìm trong toàn bộ DB xem giống ai nhất.

---

## 1. Sơ đồ tổng quan

```
                ┌────────────────────────────────────────────┐
                │           Client (Postman / Web / App)     │
                └───────────────────┬────────────────────────┘
                                    │ HTTP (multipart upload)
                                    ▼
                  ┌─────────────────────────────────────┐
                  │   Docker container: insightface-api │
                  │   Port: 8001 (host) → 8000 (cont.)  │
                  │                                     │
                  │   ┌─────────────────────────────┐   │
                  │   │  FastAPI app (main.py)      │   │
                  │   │  - Routes / Endpoints       │   │
                  │   │  - Lưu file upload          │   │
                  │   └──────┬───────────┬──────────┘   │
                  │          │           │              │
                  │          ▼           ▼              │
                  │   ┌────────────┐ ┌───────────────┐  │
                  │   │face_service│ │  database.py  │  │
                  │   │ InsightFace│ │  + models.py  │  │
                  │   │ (AI model) │ │  SQLAlchemy   │  │
                  │   └─────┬──────┘ └───────┬───────┘  │
                  └─────────┼────────────────┼──────────┘
                            ▼                ▼
                      Embedding         SQLite file
                      vector (512f)      (face.db)
                            │                │
                            ▼                ▼
                       ┌─────────────────────────┐
                       │ Volumes (host machine)  │
                       │  ./data/uploads         │
                       │  ./data/face.db         │
                       │  ./data/insightface     │
                       └─────────────────────────┘
```

---

## 2. Các thành phần (file by file)

### 2.1. `Dockerfile`
- Base image: `python:3.10-slim`
- Cài system libs: `build-essential`, `cmake`, `libgl1`, `libglib2.0-0` (cần cho OpenCV và InsightFace)
- Cài Python deps từ `requirements.txt`
- Copy thư mục `./app` vào container
- Chạy server bằng `uvicorn main:app --host 0.0.0.0 --port 8000`

### 2.2. `docker-compose.yml`
- Service `face-api`, container tên `insightface-api`
- Map port `8001 (host) → 8000 (container)` — nên truy cập qua `http://localhost:8001`
- 3 volume quan trọng để **dữ liệu không mất khi container chết**:
  - `./data/uploads` ↔ `/app/uploads` (lưu ảnh upload)
  - `./data/face.db` ↔ `/app/face.db` (file SQLite)
  - `./data/insightface` ↔ `/root/.insightface` (cache model AI ~hàng trăm MB, tránh tải lại)

### 2.3. `requirements.txt` — Các thư viện chính

| Thư viện | Vai trò |
|---|---|
| `fastapi` | Framework web để định nghĩa REST API |
| `uvicorn` | ASGI server chạy FastAPI |
| `insightface` | Thư viện AI để **detect mặt + trích xuất embedding 512 chiều** |
| `onnxruntime` | Engine chạy model ONNX (insightface dùng ONNX) |
| `opencv-python-headless` | Đọc ảnh (`cv2.imread`) |
| `numpy` | Xử lý vector embedding |
| `sqlalchemy` | ORM để thao tác DB SQLite |
| `python-multipart` | Cho phép FastAPI nhận file upload |

### 2.4. `app/database.py`
- Tạo SQLAlchemy `engine` trỏ tới file SQLite `/app/face.db`
- `SessionLocal` = factory tạo session (mỗi request mở 1 session)
- `Base` = base class cho ORM models
- `check_same_thread=False` để FastAPI (multi-thread) dùng được SQLite

### 2.5. `app/models.py` — Schema DB

Hai bảng:
- **`students`**: `id` (int, PK), `name` (string)
- **`faces`**: `id` (int, PK), `student_id` (int, FK logic), `embedding` (LargeBinary — vector float32 lưu dưới dạng bytes)

Quan hệ: 1 student ↔ nhiều face (mỗi student có thể đăng ký nhiều ảnh).

### 2.6. `app/face_service.py` — Lõi AI
- Khởi tạo `insightface.app.FaceAnalysis()`, chạy CPU (`ctx_id=-1`)
- `get_embedding(image_path)`:
  1. Đọc ảnh bằng OpenCV
  2. Detect khuôn mặt (`model.get(img)`)
  3. Trả về vector embedding của mặt đầu tiên (None nếu không có mặt)
- `cosine_similarity(a, b)`: chuẩn hoá L2 rồi dot product → giá trị từ -1 đến 1, càng gần 1 càng giống.

### 2.7. `app/main.py` — Tầng API (FastAPI)

Khai báo các endpoint:

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/` | Health check |
| POST | `/student` | Tạo student thủ công |
| POST | `/student/{id}/face` | Upload ảnh, trích embedding, lưu DB (auto tạo student nếu chưa có) |
| POST | `/verify/{student_id}` | So ảnh upload với 1 student cụ thể |
| POST | `/verify` | So ảnh với toàn DB → trả về người giống nhất |
| POST | `/verify/top` | So ảnh với toàn DB → trả về top K |
| GET | `/student/{id}/images` | Liệt kê file ảnh đã upload |
| GET | `/student/{id}/faces` | Liệt kê embedding của 1 student |
| GET | `/students` | Danh sách tất cả student |
| DELETE | `/face/{id}` | Xoá 1 embedding |
| DELETE | `/student/{id}` | Xoá student + toàn bộ face + file ảnh |

Ngoài ra: `app.mount("/uploads", ...)` để serve ảnh tĩnh — truy cập `http://localhost:8001/uploads/<file>` để xem ảnh.

---

## 3. Luồng tương tác — 2 use case quan trọng

### A. Đăng ký khuôn mặt cho 1 học sinh

```
Client gửi POST /student/123/face với file ảnh
        │
        ▼
[main.py] save_upload_file() lưu ảnh vào /app/uploads/student_123_<uuid>.jpg
        │
        ▼
[face_service] get_embedding() → cv2 đọc ảnh → InsightFace detect → vector 512 chiều
        │
        ▼ (nếu có mặt)
[main.py] tạo row Face(student_id=123, embedding=bytes(vector))
        │
        ▼
[database] SQLAlchemy commit vào face.db
        │
        ▼
Trả JSON: {face_id, image_url, ...}
```

### B. Verify (xác minh) một ảnh

```
Client gửi POST /verify với ảnh
        │
        ▼
[main.py] lấy TOÀN BỘ Face từ DB
        │
        ▼
[face_service] get_embedding() từ ảnh upload → vector "unknown"
        │
        ▼
Lặp qua từng face trong DB:
   known = np.frombuffer(f.embedding, dtype=float32)
   score = cosine_similarity(known, unknown)
   giữ score cao nhất
        │
        ▼
So sánh best_score với threshold (mặc định 0.65)
        │
        ▼
Trả JSON: {match: true/false, student_id, similarity, ...}
        │
        ▼
[finally] xoá file ảnh tạm "verify_*.jpg"
```

---

## 4. Cách vận hành thực tế (newbie guide)

```bash
cd /mnt/d/SEP490/face.db

# 1. Build & chạy
docker compose up -d --build

# 2. Lần đầu chạy: InsightFace sẽ tải model về ~/.insightface (khoảng vài trăm MB)
docker logs -f insightface-api

# 3. Mở Swagger UI
# → http://localhost:8001/docs
```

**Quy trình thử nghiệm:**
1. Vào `/docs` → thử endpoint `POST /student/1/face` → upload ảnh học sinh A
2. Upload thêm 2-3 ảnh nữa cho cùng `student_id=1` để tăng độ chính xác
3. Đăng ký thêm vài học sinh khác (`student_id=2, 3,...`)
4. Gọi `POST /verify` với 1 ảnh test → xem trả về student nào, similarity bao nhiêu
5. Điều chỉnh `threshold` (0.5 dễ match, 0.7 nghiêm ngặt hơn)

---

## 5. Điểm yếu kiến trúc (để bạn biết khi học)

- **Tìm kiếm tuyến tính O(n)**: mỗi request `/verify` đều load toàn bộ embedding rồi loop. Khi có hàng nghìn học sinh sẽ chậm. Giải pháp thật: dùng FAISS/Milvus/pgvector.
- **Model load mỗi lần khởi động**: không có lazy loading, mỗi khi container restart phải warm-up.
- **Không có authentication**: ai cũng gọi được API → chỉ phù hợp dev/internal.
- **Endpoint `verify_top` query DB N+1 lần** (loop `db.query(Student)` trong vòng lặp face) — kém hiệu suất.
- **Không có validate file type**: client có thể upload bất cứ thứ gì với đuôi `.jpg`.
- **DB khoá thread giả** (`check_same_thread=False`) — SQLite vẫn lock-write, không scale được production.

---

## 6. Tóm gọn cho dễ nhớ

> **face.db** = FastAPI + InsightFace + SQLite, đóng gói Docker.
> - **FastAPI** = lớp HTTP API
> - **InsightFace** = bộ não AI (ảnh → vector 512D)
> - **SQLAlchemy + SQLite** = nơi lưu vector dạng bytes
> - **Cosine similarity** = thước đo "giống nhau"
> - **Docker volumes** = giữ data + cache model giữa các lần chạy
