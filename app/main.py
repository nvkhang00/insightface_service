from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import numpy as np

from database import SessionLocal, Base, engine
from models import Student, Face
from face_service import get_embedding, cosine_similarity

app = FastAPI(title="InsightFace Student Recognition API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
os.makedirs(DATA_DIR, exist_ok=True)

Base.metadata.create_all(bind=engine)

ENV = os.getenv('RUN_ENV', 'local')

if ENV == 'docker':
    UPLOAD_DIR = "/app/uploads"
else:
    UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def save_upload_file(file: UploadFile, prefix: str = "") -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not ext:
        ext = ".jpg"

    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return path


def remove_file_if_exists(path: str | None) -> None:
    if path and os.path.exists(path):
        os.remove(path)


@app.get("/")
def root():
    return {"message": "Face Recognition API is running"}


# Tạo student thủ công nếu muốn
@app.post("/student")
def create_student(student_id: int, name: str):
    db = SessionLocal()
    try:
        existing = db.query(Student).filter(Student.id == student_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Student already exists")

        student = Student(id=student_id, name=name)
        db.add(student)
        db.commit()
        db.refresh(student)

        return {
            "message": "Student created",
            "student_id": int(student.id),
            "student_name": student.name
        }
    finally:
        db.close()


# Upload ảnh cho student, chưa có student thì tự tạo luôn
@app.post("/student/{student_id}/face")
async def add_face(
    student_id: int,
    file: UploadFile = File(...),
    name: str | None = None
):
    db = SessionLocal()
    image_path = None

    try:
        student = db.query(Student).filter(Student.id == student_id).first()

        if student is None:
            student = Student(
                id=student_id,
                name=name if name else f"Student {student_id}"
            )
            db.add(student)
            db.commit()
            db.refresh(student)

        image_path = save_upload_file(file, prefix=f"student_{student_id}_")

        embedding = get_embedding(image_path)
        if embedding is None:
            remove_file_if_exists(image_path)
            raise HTTPException(status_code=400, detail="No face detected")

        face = Face(
            student_id=student_id,
            embedding=embedding.astype(np.float32).tobytes()
        )
        db.add(face)
        db.commit()
        db.refresh(face)

        return {
            "message": "Face added",
            "student_id": int(student.id),
            "student_name": student.name,
            "face_id": int(face.id),
            "image_url": f"/uploads/{os.path.basename(image_path)}"
        }

    finally:
        db.close()


# Verify theo 1 student cụ thể
@app.post("/verify/{student_id}")
async def verify_by_student(
    student_id: int,
    file: UploadFile = File(...),
    threshold: float = Query(0.65, ge=0.0, le=1.0)
):
    db = SessionLocal()
    temp_path = None

    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if student is None:
            raise HTTPException(status_code=404, detail="Student not found")

        faces = db.query(Face).filter(Face.student_id == student_id).all()
        if not faces:
            raise HTTPException(status_code=400, detail="Student has no registered face data")

        temp_path = save_upload_file(file, prefix="verify_")

        unknown = get_embedding(temp_path)
        if unknown is None:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")

        best_score = -1.0

        for f in faces:
            known = np.frombuffer(f.embedding, dtype=np.float32)
            score = float(cosine_similarity(known, unknown))

            if score > best_score:
                best_score = score

        return {
            "match": bool(best_score >= threshold),
            "student_id": int(student.id),
            "student_name": student.name,
            "similarity": float(best_score),
            "threshold": float(threshold)
        }

    finally:
        remove_file_if_exists(temp_path)
        db.close()


# Verify toàn bộ DB, trả về người giống nhất
@app.post("/verify")
async def verify_all(
    file: UploadFile = File(...),
    threshold: float = Query(0.65, ge=0.0, le=1.0)
):
    db = SessionLocal()
    temp_path = None

    try:
        all_faces = db.query(Face).all()
        if not all_faces:
            raise HTTPException(status_code=400, detail="No registered face data in database")

        temp_path = save_upload_file(file, prefix="verify_")

        unknown = get_embedding(temp_path)
        if unknown is None:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")

        best_score = -1.0
        best_student_id = None

        for f in all_faces:
            known = np.frombuffer(f.embedding, dtype=np.float32)
            score = float(cosine_similarity(known, unknown))

            if score > best_score:
                best_score = score
                best_student_id = f.student_id

        if best_student_id is None:
            raise HTTPException(status_code=404, detail="No matching student found")

        student = db.query(Student).filter(Student.id == best_student_id).first()

        return {
            "match": bool(best_score >= threshold),
            "student_id": int(best_student_id),
            "student_name": student.name if student else None,
            "similarity": float(best_score),
            "threshold": float(threshold)
        }

    finally:
        remove_file_if_exists(temp_path)
        db.close()


# Verify toàn bộ DB, trả về top K người giống nhất
@app.post("/verify/top")
async def verify_top(
    file: UploadFile = File(...),
    threshold: float = Query(0.65, ge=0.0, le=1.0),
    top_k: int = Query(5, ge=1, le=20)
):
    db = SessionLocal()
    temp_path = None

    try:
        all_faces = db.query(Face).all()
        if not all_faces:
            raise HTTPException(status_code=400, detail="No registered face data in database")

        temp_path = save_upload_file(file, prefix="verify_")

        unknown = get_embedding(temp_path)
        if unknown is None:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")

        raw_results = []

        for f in all_faces:
            known = np.frombuffer(f.embedding, dtype=np.float32)
            score = float(cosine_similarity(known, unknown))

            student = db.query(Student).filter(Student.id == f.student_id).first()

            raw_results.append({
                "student_id": int(f.student_id),
                "student_name": student.name if student else None,
                "similarity": float(score)
            })

        # Nếu 1 student có nhiều ảnh thì chỉ lấy score cao nhất
        best_by_student = {}
        for item in raw_results:
            sid = item["student_id"]
            if sid not in best_by_student or item["similarity"] > best_by_student[sid]["similarity"]:
                best_by_student[sid] = item

        final_results = sorted(
            best_by_student.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]

        best_match = final_results[0] if final_results else None

        return {
            "match": bool(best_match and best_match["similarity"] >= threshold),
            "threshold": float(threshold),
            "best_match": best_match,
            "top_k": final_results
        }

    finally:
        remove_file_if_exists(temp_path)
        db.close()


# Lấy danh sách ảnh của 1 học sinh
@app.get("/student/{student_id}/images")
def get_images(student_id: int):
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if student is None:
            raise HTTPException(status_code=404, detail="Student not found")

        prefix = f"student_{student_id}_"
        images = []

        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(prefix):
                images.append({
                    "file_name": filename,
                    "url": f"/uploads/{filename}"
                })

        return {
            "student_id": int(student.id),
            "student_name": student.name,
            "total_images": len(images),
            "images": images
        }

    finally:
        db.close()


# Lấy danh sách face embedding của 1 học sinh
@app.get("/student/{student_id}/faces")
def get_faces(student_id: int):
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if student is None:
            raise HTTPException(status_code=404, detail="Student not found")

        faces = db.query(Face).filter(Face.student_id == student_id).all()

        return {
            "student_id": int(student.id),
            "student_name": student.name,
            "total_faces": len(faces),
            "faces": [
                {
                    "face_id": int(f.id)
                }
                for f in faces
            ]
        }

    finally:
        db.close()


# Lấy danh sách student
@app.get("/students")
def get_students():
    db = SessionLocal()
    try:
        students = db.query(Student).all()

        result = []
        for s in students:
            total_faces = db.query(Face).filter(Face.student_id == s.id).count()
            result.append({
                "student_id": int(s.id),
                "student_name": s.name,
                "total_faces": int(total_faces)
            })

        return result

    finally:
        db.close()


# Xóa 1 face embedding
@app.delete("/face/{face_id}")
def delete_face(face_id: int):
    db = SessionLocal()
    try:
        face = db.query(Face).filter(Face.id == face_id).first()
        if face is None:
            raise HTTPException(status_code=404, detail="Face not found")

        db.delete(face)
        db.commit()

        return {
            "message": "Face deleted",
            "face_id": int(face_id)
        }

    finally:
        db.close()


# Xóa 1 student và toàn bộ face của student đó
@app.delete("/student/{student_id}")
def delete_student(student_id: int):
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if student is None:
            raise HTTPException(status_code=404, detail="Student not found")

        db.query(Face).filter(Face.student_id == student_id).delete()
        db.delete(student)
        db.commit()

        # xóa file ảnh đã upload của student đó
        prefix = f"student_{student_id}_"
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(prefix):
                remove_file_if_exists(os.path.join(UPLOAD_DIR, filename))

        return {
            "message": "Student deleted",
            "student_id": int(student_id)
        }

    finally:
        db.close()