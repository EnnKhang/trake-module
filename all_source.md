Directory structure:
└── AIC2026/
    ├── mock_data.ipynb
    ├── testdatakeyframe.ipynb
    ├── trake_branch.ipynb
    ├── trake_retriever.py
    ├── aic_agent_core/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── config.py
    │   ├── exceptions.py
    │   ├── prompts.py
    │   ├── query_models.py
    │   └── router.py
    └── data/
        ├── clip_features/
        │   ├── video_01.npy
        │   ├── video_02.npy
        │   └── video_03.npy
        └── csv_metadata/
            ├── video_01.csv
            ├── video_02.csv
            └── video_03.csv

================================================
FILE: mock_data.ipynb
================================================
# Jupyter notebook converted to Python script.

import os
import numpy as np
import pandas as pd

# Đường dẫn 2 thư mục riêng biệt
CLIP_DIR_MOCK = "data/clip_features"
CSV_DIR_MOCK = "data/csv_metadata"

os.makedirs(CLIP_DIR_MOCK, exist_ok=True)
os.makedirs(CSV_DIR_MOCK, exist_ok=True)

mock_videos = {
    "video_01": 150,
    "video_02": 200,
    "video_03": 120,
}

EMBEDDING_DIM = 512

for video_id, num_frames in mock_videos.items():
    # 1. Lưu file .npy vào thư mục clip_features
    features = np.random.randn(num_frames, EMBEDDING_DIM).astype(np.float32)
    np.save(os.path.join(CLIP_DIR_MOCK, f"{video_id}.npy"), features)
    
    # 2. Lưu file .csv vào thư mục csv_metadata
    df = pd.DataFrame({
        "frame_idx": range(num_frames),
        "timestamp": [round(i * 0.2, 2) for i in range(num_frames)]
    })
    df.to_csv(os.path.join(CSV_DIR_MOCK, f"{video_id}.csv"), index=False)

print(f"Đã tạo file npy tại: {os.listdir(CLIP_DIR_MOCK)}")
print(f"Đã tạo file csv tại: {os.listdir(CSV_DIR_MOCK)}")
# Output:
#   Đã tạo file npy tại: ['video_01.npy', 'video_02.npy', 'video_03.npy']

#   Đã tạo file csv tại: ['video_01.csv', 'video_02.csv', 'video_03.csv']




================================================
FILE: testdatakeyframe.ipynb
================================================
# Jupyter notebook converted to Python script.

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip

# Cấu hình
IMAGE_DIR = "data/raw_keyframes/video_01"
OUTPUT_NPY = "data/clip_features/video_01.npy"
OUTPUT_CSV = "data/csv_metadata/video_01.csv"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

os.makedirs("data/clip_features", exist_ok=True)
os.makedirs("data/csv_metadata", exist_ok=True)

# 1. Load CLIP Model & Preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(device).eval()

# 2. Đọc danh sách ảnh theo thứ tự
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

embeddings = []
metadata = []

print(f"Đang trích xuất đặc trưng cho {len(image_files)} keyframes...")

with torch.no_grad():
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, fname)
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        
        # Trích xuất image vector & chuẩn hóa L2
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy()[0])
        
        # Giả lập timestamp (ví dụ mỗi keyframe cách nhau ~4 giây theo đồng hồ bản tin)
        metadata.append({
            "frame_idx": idx + 1,
            "filename": fname,
            "pts_time": idx * 4.0
        })

# 3. Lưu file .npy và .csv
embeddings = np.array(embeddings, dtype=np.float32)
np.save(OUTPUT_NPY, embeddings)
pd.DataFrame(metadata).to_csv(OUTPUT_CSV, index=False)

print(f"✅ Đã lưu vector: {OUTPUT_NPY} (Shape: {embeddings.shape})")
print(f"✅ Đã lưu metadata: {OUTPUT_CSV}")
# Output:
#   Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

#   WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

#   Đang trích xuất đặc trưng cho 307 keyframes...

#   ✅ Đã lưu vector: data/clip_features/video_01.npy (Shape: (307, 512))

#   ✅ Đã lưu metadata: data/csv_metadata/video_01.csv




================================================
FILE: trake_branch.ipynb
================================================
# Jupyter notebook converted to Python script.

import os
import sys
import numpy as np
import pandas as pd
from pymilvus import MilvusClient, DataType

# Đường dẫn dữ liệu
CLIP_DIR = "data/clip_features"
CSV_DIR = "data/csv_metadata"
MILVUS_URI = "./milvus_demo.db"     # Đổi thành "http://localhost:19530" nếu dùng Docker
COLLECTION_NAME = "clip_keyframes"

def normalize(vectors: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 cho mảng vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms

def create_collection(client: MilvusClient, dim: int, fresh: bool = False):
    """Khởi tạo collection với số chiều vector được truyền động."""
    if client.has_collection(COLLECTION_NAME):
        if fresh:
            print(f"[-] Xóa collection cũ: {COLLECTION_NAME}")
            client.drop_collection(COLLECTION_NAME)
        else:
            return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="frame_id", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)

    # Cấu hình Index: Dùng FLAT cho Milvus Lite local để không đòi faiss-cpu
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",          # Đổi thành "HNSW" nếu chạy server Milvus lớn
        metric_type="IP",           # Inner Product (Cosine similarity khi vector đã L2-normalized)
        params={},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"[+] Đã tạo collection mới: {COLLECTION_NAME} (dim={dim})")

def get_existing_videos(client: MilvusClient) -> set:
    """Lấy danh sách video_id đã có trong collection để tránh nạp trùng."""
    existing = set()
    try:
        iterator = client.query_iterator(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["video_id"],
            batch_size=1000,
        )
        while True:
            batch = iterator.next()
            if not batch:
                break
            existing.update(row["video_id"] for row in batch)
        iterator.close()
    except Exception:
        pass
    return existing

def insert_video(client: MilvusClient, npy_file: str) -> int:
    """Nạp vector và metadata của 1 video vào Milvus."""
    video_id = os.path.splitext(npy_file)[0]
    npy_path = os.path.join(CLIP_DIR, npy_file)
    csv_path = os.path.join(CSV_DIR, f"{video_id}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy CSV tương ứng: {csv_path}")

    # Đọc vector
    features = np.load(npy_path).astype(np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Đọc CSV metadata và xử lý frame_idx linh hoạt
    df = pd.read_csv(csv_path)
    if "frame_idx" in df.columns:
        frame_ids = df["frame_idx"].astype(int).tolist()
    elif "frame_id" in df.columns:
        frame_ids = df["frame_id"].astype(int).tolist()
    else:
        frame_ids = list(range(1, len(df) + 1))

    if len(features) != len(frame_ids):
        raise ValueError(
            f"{video_id}: Lệch kích thước ({len(features)} vectors != {len(frame_ids)} frames)"
        )

    features = normalize(features)

    rows = [
        {
            "video_id": video_id,
            "frame_id": frame_ids[i],
            "embedding": features[i].tolist(),
        }
        for i in range(len(features))
    ]

    result = client.insert(collection_name=COLLECTION_NAME, data=rows)
    print(f"  + {video_id}: Đã nạp {result['insert_count']} vectors")
    return result["insert_count"]

def run(mode: str = "build"):
    if not os.path.exists(CLIP_DIR):
        raise FileNotFoundError(f"Thư mục '{CLIP_DIR}' không tồn tại.")

    npy_files = sorted(f for f in os.listdir(CLIP_DIR) if f.endswith(".npy"))
    if not npy_files:
        raise RuntimeError(f"Không tìm thấy file .npy nào trong thư mục '{CLIP_DIR}'")

    # Tự động lấy số chiều vector từ file npy đầu tiên
    first_npy = np.load(os.path.join(CLIP_DIR, npy_files[0]))
    dim = first_npy.shape[-1]

    client = MilvusClient(uri=MILVUS_URI)
    create_collection(client, dim=dim, fresh=(mode == "build"))

    existing_videos = get_existing_videos(client) if mode == "rebuild" else set()

    total, added, skipped = 0, 0, 0
    print(f"\n--- BẮT ĐẦU NẠP DỮ LIỆU (Mode: {mode}) ---")
    for npy_file in npy_files:
        video_id = os.path.splitext(npy_file)[0]
        if video_id in existing_videos:
            print(f"  - Skip: {video_id} (Đã có sẵn)")
            skipped += 1
            continue
        total += insert_video(client, npy_file)
        added += 1

    # Nạp collection vào bộ nhớ để sẵn sàng search
    client.flush(collection_name=COLLECTION_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    print(f"\n Hoàn tất nạp dữ liệu!")
    print(f"Tổng video nạp mới : {added} ({total} vectors)")
    if mode == "rebuild":
        print(f"Tổng video bỏ qua  : {skipped}")

if __name__ == "__main__":
    run(mode="build")
# Output:
#   ERROR:grpc._server:Exception calling application: Method not implemented!

#   Traceback (most recent call last):

#     File "c:\conda\Miniconda3\envs\vlm\Lib\site-packages\grpc\_server.py", line 608, in _call_behavior

#       response_or_iterator = behavior(argument, context)

#                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

#     File "c:\conda\Miniconda3\envs\vlm\Lib\site-packages\pymilvus\grpc_gen\milvus_pb2_grpc.py", line 1264, in AllocTimestamp

#       raise NotImplementedError('Method not implemented!')

#   NotImplementedError: Method not implemented!

#   [-] Xóa collection cũ: clip_keyframes

#   [+] Đã tạo collection mới: clip_keyframes (dim=512)

#   

#   --- BẮT ĐẦU NẠP DỮ LIỆU (Mode: build) ---

#     + video_01: Đã nạp 307 vectors

#     + video_02: Đã nạp 200 vectors

#     + video_03: Đã nạp 120 vectors

#   

#    Hoàn tất nạp dữ liệu!

#   Tổng video nạp mới : 3 (627 vectors)


import numpy as np
from trake_retriever import TrakeEngine

# Truyền rõ đường dẫn file database và collection đã nạp ở Cell 1
engine = TrakeEngine(
    milvus_uri="./milvus_demo.db",
    collection_name="clip_keyframes"
)

# Kiểm tra encode thử 1 câu
emb = engine._encode_text("a man kicking a soccer ball")
print("Embedding shape:", emb.shape)
print("Norm xấp xỉ 1.0:", np.linalg.norm(emb))
# Output:
#   Error: MilvusException: <MilvusException: (code=2, message=Fail connecting to server on localhost:19530, illegal connection params or server unavailable)>

from aic_agent_core import route_query, TaskType
from trake_retriever import TrakeEngine

# 1. Khởi tạo engine trỏ đúng database và thư mục dữ liệu ở Cell 1
engine = TrakeEngine(
    milvus_uri="./milvus_demo.db",
    collection_name="clip_keyframes",
    csv_dir="data/csv_metadata",
    feature_dir="data/clip_features"
)

raw_query = (
    "First is the sunset city skyline with 60 seconds news logo, "
    "then two news anchors male and female presenting in newsroom studio, "
    "after that an aerial drone shot of collapsed asphalt road falling into river water, "
    "and finally a white car driving into hospital entrance."
)

# 3. Router phân tách query
task_type, structured_query = route_query(raw_query)
print(f"Task Type: {task_type}\n")

if task_type == TaskType.TRAKE:
    # 4. Tìm kiếm video và căn chỉnh chuỗi frame
    best_video, keyframe_ids = engine.solve_trake(
        structured_query, 
        top_k_videos=5, 
        min_gap=1
    )

    print("================ KẾT QUẢ TRAKE ================")
    print(f"Video khớp nhất: {best_video}")
    print(f"Danh sách Frame IDs tìm được: {keyframe_ids}\n")
    
    if keyframe_ids:
        for i, (ev, f_id) in enumerate(zip(structured_query.events, keyframe_ids), 1):
            print(f"  📌 Event {i}: \"{ev.description}\" ➔ Khớp Frame ID: {f_id} (ảnh {str(f_id).zfill(3)}.jpg)")
    else:
        print(" Không tìm thấy chuỗi frame thỏa mãn ràng buộc thời gian.")
else:
    print(f"Query được phân loại sang {task_type}, không phải TRAKE.")
# Output:
#   c:\conda\Miniconda3\envs\vlm\Lib\site-packages\open_clip\factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True).

#     warnings.warn(

#   Task Type: TRAKE

#   

#   ================ KẾT QUẢ TRAKE ================

#   Video khớp nhất: video_01

#   Danh sách Frame IDs tìm được: [1, 12, 30, 67]

#   

#     📌 Event 1: "Sunset city skyline showing the 60 seconds news logo" ➔ Khớp Frame ID: 1 (ảnh 001.jpg)

#     📌 Event 2: "Two news anchors, one male and one female, presenting in a newsroom studio" ➔ Khớp Frame ID: 12 (ảnh 012.jpg)

#     📌 Event 3: "Aerial drone shot of a collapsed asphalt road falling into river water" ➔ Khớp Frame ID: 30 (ảnh 030.jpg)

#     📌 Event 4: "A white car driving into the hospital entrance" ➔ Khớp Frame ID: 67 (ảnh 067.jpg)




================================================
FILE: trake_retriever.py
================================================
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from pymilvus import MilvusClient

# Cấu hình mặc định (tùy chỉnh lại nếu đường dẫn dự án khác)
MILVUS_URI = "./milvus_demo.db"
COLLECTION_NAME = "clip_keyframes"
CSV_DIR = "data/csv_metadata"
EMBEDDING_DIM = 512


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm == 0, v, v / norm)


class TrakeEngine:
    def __init__(
        self,
        milvus_uri: str = MILVUS_URI,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = COLLECTION_NAME
        self.client.load_collection(self.collection_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Bộ nhớ đệm tránh tính toán lại nhiều lần
        self._text_embed_cache: dict[str, np.ndarray] = {}
        self._feature_cache: dict[str, tuple[np.ndarray, list[int]]] = {}

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode văn bản sang vector embedding 512d qua OpenCLIP và lưu cache."""
        if text in self._text_embed_cache:
            return self._text_embed_cache[text]

        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            feat = self.clip_model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat.cpu().numpy()[0].astype(np.float32)

        self._text_embed_cache[text] = vec
        return vec

    def retrieve_candidate_videos(self, structured_query, top_k: int = 5) -> list[str]:
        """Truy vấn Milvus theo từng sub-event và cộng dồn điểm (vote) để xếp hạng video."""
        video_scores: dict[str, float] = {}

        for ev in structured_query.events:
            query_vec = self._encode_text(ev.description)
            search_res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vec.tolist()],
                anns_field="embedding",
                search_params={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=50,
                output_fields=["video_id"],
            )
            if search_res and len(search_res) > 0:
                for hit in search_res[0]:
                    vid = hit["entity"]["video_id"]
                    video_scores[vid] = video_scores.get(vid, 0.0) + hit["distance"]

        ranked = sorted(video_scores.items(), key=lambda x: -x[1])
        return [vid for vid, _ in ranked[:top_k]]

    def _load_video_features(self, video_id: str):
        """Đọc và chuẩn hóa vector đặc trưng npy + danh sách frame_idx từ CSV."""
        if video_id in self._feature_cache:
            return self._feature_cache[video_id]

        csv_file = os.path.join(CSV_DIR, f"{video_id}.csv")
        feature_dir = "data/clip_features" if os.path.exists("data/clip_features") else "data"
        npy_file = os.path.join(feature_dir, f"{video_id}.npy")

        if not os.path.exists(csv_file) or not os.path.exists(npy_file):
            return None, None

        df = pd.read_csv(csv_file)
        frame_ids = df["frame_idx"].tolist()
        features = np.load(npy_file).astype(np.float32)
        features = normalize(features)

        self._feature_cache[video_id] = (features, frame_ids)
        return features, frame_ids

    def solve_trake(self, structured_query, top_k_videos: int = 5, min_gap: int = 1):
        """Quy hoạch động tìm chuỗi frame khớp tuần tự nhất với các sub-events."""
        events = structured_query.events
        N = len(events)
        candidate_vids = self.retrieve_candidate_videos(structured_query, top_k=top_k_videos)

        if not candidate_vids:
            return None, []

        best_video_id = None
        best_score = -float("inf")
        best_frames = []

        # Vector embedding cho từng sub-event
        event_embeddings = np.stack([self._encode_text(ev.description) for ev in events])

        for vid in candidate_vids:
            features, frame_ids = self._load_video_features(vid)
            if features is None or len(features) < N:
                continue

            T = len(features)
            # Ma trận tương đồng Cosine giữa N sub-events và T frames
            sim_matrix = np.dot(event_embeddings, features.T)  # Shape: (N, T)

            # Khởi tạo bảng DP và Backtrack
            dp = np.full((N, T), -np.inf, dtype=np.float32)
            backtrack = np.zeros((N, T), dtype=np.int32)

            dp[0, :] = sim_matrix[0, :]

            for i in range(1, N):
                for t in range(i * min_gap, T):
                    prev_valid_t = range(0, t - min_gap + 1)
                    if not prev_valid_t:
                        continue
                    prev_scores = dp[i - 1, prev_valid_t]
                    best_prev_idx = int(np.argmax(prev_scores))
                    best_prev_score = prev_scores[best_prev_idx]

                    if best_prev_score != -np.inf:
                        dp[i, t] = best_prev_score + sim_matrix[i, t]
                        backtrack[i, t] = best_prev_idx

            last_t = int(np.argmax(dp[N - 1, :]))
            current_score = dp[N - 1, last_t]

            if current_score > best_score and current_score != -np.inf:
                best_score = current_score
                best_video_id = vid

                # Truy vết đường đi tối ưu
                path = [last_t]
                curr = last_t
                for i in range(N - 1, 0, -1):
                    curr = backtrack[i, curr]
                    path.append(curr)
                path.reverse()
                best_frames = [frame_ids[idx] for idx in path]

        if not best_frames:
            return None, []
        return best_video_id, best_frames


================================================
FILE: aic_agent_core/__init__.py
================================================
"""Public API for the AIC 2026 query-understanding agent core."""

from .query_models import (
    AnswerType,
    CommonQuery,
    Entity,
    KISQuery,
    LLMStructuredQuery,
    QAQuery,
    RouteResult,
    StructuredQuery,
    TRAKEQuery,
    TaskType,
    TemporalEvent,
)
from .router import QueryRouter, aroute_query, route_query

__all__ = [
    "AnswerType",
    "CommonQuery",
    "Entity",
    "KISQuery",
    "LLMStructuredQuery",
    "QAQuery",
    "QueryRouter",
    "RouteResult",
    "StructuredQuery",
    "TaskType",
    "TemporalEvent",
    "TRAKEQuery",
    "aroute_query",
    "route_query",
]



================================================
FILE: aic_agent_core/__main__.py
================================================
from __future__ import annotations

import json
import sys

from .router import route_query


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("AIC 2026 Query Router (Ctrl+C Ä‘á»ƒ thoÃ¡t)")
    raw_query = input("Nháº­p truy váº¥n: ").strip()
    query_id = input("Query ID (Enter Ä‘á»ƒ tá»± sinh): ").strip() or None
    task_type, structured_query = route_query(raw_query, query_id=query_id)
    print(f"\nTask type: {task_type.value}")
    print(json.dumps(structured_query.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



================================================
FILE: aic_agent_core/config.py
================================================
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    """Runtime settings. Values can be supplied through environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AIC_",
        env_file=".env",
        env_file_encoding="utf-8",  
        extra="ignore",
    )

    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=8192, ge=512, le=65536)
    gemini_max_attempts: int = Field(default=2, ge=1, le=4)




================================================
FILE: aic_agent_core/exceptions.py
================================================
class QueryRouterError(RuntimeError):
    """Base error raised by the query router."""


class EmptyQueryError(QueryRouterError, ValueError):
    """Raised when the raw query is empty or only whitespace."""


class RouterConfigurationError(QueryRouterError):
    """Raised when Gemini credentials or settings are invalid."""


class StructuredOutputError(QueryRouterError):
    """Raised when Gemini repeatedly returns an invalid structured result."""




================================================
FILE: aic_agent_core/prompts.py
================================================
SYSTEM_PROMPT = """
You are the query-understanding and routing core for the AIC 2026 video retrieval
system. Inputs may be Vietnamese, English, or mixed. Preserve concrete details
and never invent facts that are not stated or strongly entailed.

Classify using these competition definitions:
- KIS: locate one described event and return a frame from it.
- QA: locate an event AND answer an explicit or clearly implied question about it.
- TRAKE: locate one video, then align two or more distinct semantic moments in a
  chronological event chain. A long description with many simultaneous visual
  attributes is still KIS, not TRAKE.

Parsing requirements:
1. Resolve every pronoun or elliptical reference into a standalone phrase.
2. Extract searchable people, objects, actions, scenes, text, and their attributes
   into entities. Set needs_ocr only when visible text is needed, and needs_asr only
   when speech or audio is needed.
3. Produce concise Vietnamese and English query_variants. visual_description must
   contain visual evidence only; do not put a guessed QA answer in it.
4. For QA, keep the retrieval context separate from the question and infer the
   expected answer type. Retrieval queries must not leak a guessed answer.
5. For TRAKE, create events in exact chronological order. Each event description
   must stand alone and each semantic_keyframe must define the precise instant to
   align (onset, contact, peak, completion, etc.). temporal_constraints must state
   the chronological relations, such as "event 1 before event 2".
6. Return one flat object. KIS has question=null, answer_type=null, and empty
   events/temporal_constraints. QA has question and answer_type, with empty
   events/temporal_constraints. TRAKE has question=null, answer_type=null, at
   least two ordered events, and temporal_constraints.
7. query_id and raw_query must exactly reproduce the supplied values.
""".strip()


def build_user_prompt(
    query_id: str,
    raw_text: str,
    validation_feedback: str | None = None,
) -> str:
    prompt = (
        "Analyze and route this AIC 2026 query:\n\n"
        f"<query_id>{query_id}</query_id>\n"
        f"<query>\n{raw_text}\n</query>"
    )
    if validation_feedback:
        prompt += (
            "\n\nYour previous output failed application validation. Correct the semantic "
            "structure while analyzing the same query. Validation feedback:\n"
            f"{validation_feedback}"
        )
    return prompt



================================================
FILE: aic_agent_core/query_models.py
================================================
from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TaskType(StrEnum):
    KIS = "KIS"
    QA = "QA"
    TRAKE = "TRAKE"


class AnswerType(StrEnum):
    COUNT = "count"
    COLOR = "color"
    TEXT = "text"
    ENTITY = "entity"
    BOOLEAN = "boolean"
    LOCATION = "location"
    TIME = "time"
    ACTION = "action"
    OTHER = "other"


class Entity(StrictModel):
    type: str = Field(min_length=1, description="person, object, action, scene, or text")
    value: str = Field(min_length=1, description="Normalized searchable value")
    attribute: str | None = Field(default=None, description="Color, quantity, clothing, or relation")


class TemporalEvent(StrictModel):
    index: int = Field(ge=1)
    description: str = Field(min_length=3, description="Standalone resolved event")
    semantic_keyframe: str = Field(min_length=3, description="Exact instant to align")
    query_variants: list[str] = Field(min_length=1, max_length=4)
    entities: list[Entity] = Field(default_factory=list)


class CommonQuery(StrictModel):
    query_id: str = Field(min_length=1)
    raw_query: str = Field(min_length=1)
    query_variants: list[str] = Field(min_length=2, max_length=8)
    visual_description: str = Field(min_length=3)
    entities: list[Entity] = Field(default_factory=list)
    needs_ocr: bool
    needs_asr: bool


class KISQuery(CommonQuery):
    question: None = None
    events: list[TemporalEvent] = Field(default_factory=list, max_length=0)
    temporal_constraints: list[str] = Field(default_factory=list, max_length=0)


class QAQuery(CommonQuery):
    question: str = Field(min_length=2)
    answer_type: AnswerType
    events: list[TemporalEvent] = Field(default_factory=list, max_length=0)
    temporal_constraints: list[str] = Field(default_factory=list, max_length=0)


class TRAKEQuery(CommonQuery):
    question: None = None
    events: list[TemporalEvent] = Field(min_length=2, max_length=20)
    temporal_constraints: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_event_order(self) -> "TRAKEQuery":
        indexes = [event.index for event in self.events]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("TRAKE event indexes must be consecutive and chronological")
        return self


StructuredQuery: TypeAlias = KISQuery | QAQuery | TRAKEQuery


class LLMStructuredQuery(CommonQuery):
    """Flat schema sent to Gemini, then converted to a public query class."""

    task_type: TaskType
    question: str | None = None
    answer_type: AnswerType | None = None
    events: list[TemporalEvent] = Field(default_factory=list, max_length=20)
    temporal_constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_fields(self) -> "LLMStructuredQuery":
        if self.task_type is TaskType.KIS:
            if self.question is not None or self.answer_type is not None:
                raise ValueError("KIS requires question=null and answer_type=null")
            if self.events or self.temporal_constraints:
                raise ValueError("KIS requires events=[] and temporal_constraints=[]")
        elif self.task_type is TaskType.QA:
            if not self.question or self.answer_type is None:
                raise ValueError("QA requires question and answer_type")
            if self.events or self.temporal_constraints:
                raise ValueError("QA requires events=[] and temporal_constraints=[]")
        else:
            if self.question is not None or self.answer_type is not None:
                raise ValueError("TRAKE requires question=null and answer_type=null")
            if len(self.events) < 2 or not self.temporal_constraints:
                raise ValueError("TRAKE requires ordered events and temporal_constraints")
            indexes = [event.index for event in self.events]
            if indexes != list(range(1, len(indexes) + 1)):
                raise ValueError("TRAKE event indexes must be consecutive and chronological")
        return self

    def to_public_query(self) -> StructuredQuery:
        common = self.model_dump(
            exclude={"task_type", "question", "answer_type", "events", "temporal_constraints"}
        )
        if self.task_type is TaskType.KIS:
            return KISQuery(**common)
        if self.task_type is TaskType.QA:
            assert self.question is not None and self.answer_type is not None
            return QAQuery(**common, question=self.question, answer_type=self.answer_type)
        return TRAKEQuery(
            **common,
            events=self.events,
            temporal_constraints=self.temporal_constraints,
        )


class RouteResult(StrictModel):
    task_type: TaskType
    structured_query: StructuredQuery



================================================
FILE: aic_agent_core/router.py
================================================
from __future__ import annotations

from uuid import uuid4
from typing import Any

from pydantic import ValidationError

from .config import RouterSettings
from .exceptions import (
    EmptyQueryError,
    RouterConfigurationError,
    StructuredOutputError,
)
from .query_models import LLMStructuredQuery, RouteResult, StructuredQuery, TaskType
from .prompts import SYSTEM_PROMPT, build_user_prompt


class QueryRouter:
    """Gemini-backed semantic parser and AIC task router.

    ``client`` is injectable to make the core deterministic in unit tests.
    """

    def __init__(self, settings: RouterSettings | None = None, client: Any | None = None):
        self.settings = settings or RouterSettings()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.google_api_key:
                raise RouterConfigurationError(
                    "GOOGLE_API_KEY is missing. Put it in the environment or .env file."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise RouterConfigurationError(
                    "google-genai is not installed; run `pip install -e .`"
                ) from exc
            self._client = genai.Client(api_key=self.settings.google_api_key)
        return self._client

    def route(self, raw_text: str, query_id: str | None = None) -> RouteResult:
        text = _validate_raw_text(raw_text)
        resolved_query_id = _resolve_query_id(query_id)
        feedback: str | None = None
        last_error: Exception | None = None

        for _ in range(self.settings.gemini_max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=build_user_prompt(resolved_query_id, text, feedback),
                    config=self._generation_config(),
                )
                llm_query = self._parse_response(response)
                query = llm_query.to_public_query()
                self._validate_against_input(query, resolved_query_id, text)
                return RouteResult(task_type=llm_query.task_type, structured_query=query)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                feedback = _compact_error(exc)

        raise StructuredOutputError(
            f"Gemini returned invalid structured output after "
            f"{self.settings.gemini_max_attempts} attempt(s): {last_error}"
        ) from last_error

    async def aroute(self, raw_text: str, query_id: str | None = None) -> RouteResult:
        text = _validate_raw_text(raw_text)
        resolved_query_id = _resolve_query_id(query_id)
        feedback: str | None = None
        last_error: Exception | None = None

        for _ in range(self.settings.gemini_max_attempts):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=build_user_prompt(resolved_query_id, text, feedback),
                    config=self._generation_config(),
                )
                llm_query = self._parse_response(response)
                query = llm_query.to_public_query()
                self._validate_against_input(query, resolved_query_id, text)
                return RouteResult(task_type=llm_query.task_type, structured_query=query)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                feedback = _compact_error(exc)

        raise StructuredOutputError(
            f"Gemini returned invalid structured output after "
            f"{self.settings.gemini_max_attempts} attempt(s): {last_error}"
        ) from last_error

    def route_query(
        self, raw_text: str, query_id: str | None = None
    ) -> tuple[TaskType, StructuredQuery]:
        result = self.route(raw_text, query_id)
        return result.task_type, result.structured_query

    async def aroute_query(
        self, raw_text: str, query_id: str | None = None
    ) -> tuple[TaskType, StructuredQuery]:
        result = await self.aroute(raw_text, query_id)
        return result.task_type, result.structured_query

    def _generation_config(self) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RouterConfigurationError(
                "google-genai is not installed; run `pip install -e .`"
            ) from exc
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=self.settings.gemini_temperature,
            max_output_tokens=self.settings.gemini_max_output_tokens,
            response_mime_type="application/json",
            # Pass JSON Schema directly instead of ``response_schema=StructuredQuery``.
            # The SDK's typed-schema conversion serializes Pydantic's
            # ``additionalProperties`` as ``additional_properties``, which the
            # Gemini generateContent endpoint rejects with HTTP 400.
            response_json_schema=_gemini_json_schema(),
        )

    @staticmethod
    def _parse_response(response: Any) -> LLMStructuredQuery:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, LLMStructuredQuery):
            return parsed
        if parsed is not None:
            return LLMStructuredQuery.model_validate(parsed)
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Gemini response contains neither parsed data nor text")
        return LLMStructuredQuery.model_validate_json(response_text)

    @staticmethod
    def _validate_against_input(
        query: StructuredQuery, query_id: str, raw_text: str
    ) -> None:
        if query.query_id != query_id:
            raise ValueError("query_id must exactly match the supplied query_id")
        if query.raw_query != raw_text:
            raise ValueError("raw_query must exactly match the raw input")


_default_router: QueryRouter | None = None


def _get_default_router() -> QueryRouter:
    global _default_router
    if _default_router is None:
        _default_router = QueryRouter()
    return _default_router


def route_query(
    raw_text: str, query_id: str | None = None
) -> tuple[TaskType, StructuredQuery]:
    """Route one raw query using a lazily initialized default Gemini client."""

    return _get_default_router().route_query(raw_text, query_id)


async def aroute_query(
    raw_text: str, query_id: str | None = None
) -> tuple[TaskType, StructuredQuery]:
    """Async counterpart of :func:`route_query`."""

    return await _get_default_router().aroute_query(raw_text, query_id)


def _validate_raw_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    text = raw_text.strip()
    if not text:
        raise EmptyQueryError("raw_text must not be empty")
    return text


def _resolve_query_id(query_id: str | None) -> str:
    if query_id is None:
        return f"q_{uuid4().hex[:12]}"
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("query_id must be a non-empty string")
    return query_id.strip()


def _compact_error(error: Exception, limit: int = 1800) -> str:
    value = str(error).replace("\x00", "")
    return value[:limit]


def _gemini_json_schema() -> dict[str, Any]:
    """Return Gemini-compatible JSON Schema without weakening local validation.

    ``extra='forbid'`` remains active in Pydantic when validating Gemini's
    response. Only the unsupported schema keyword is removed from the API
    request payload.
    """

    schema = LLMStructuredQuery.model_json_schema()
    return _remove_schema_key(schema, "additionalProperties")


def _remove_schema_key(value: Any, key_to_remove: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_schema_key(child, key_to_remove)
            for key, child in value.items()
            if key != key_to_remove
        }
    if isinstance(value, list):
        return [_remove_schema_key(child, key_to_remove) for child in value]
    return value



================================================
FILE: data/clip_features/video_01.npy
================================================
[Binary file]


================================================
FILE: data/clip_features/video_02.npy
================================================
[Binary file]


================================================
FILE: data/clip_features/video_03.npy
================================================
[Binary file]


================================================
FILE: data/csv_metadata/video_01.csv
================================================
frame_idx,filename,pts_time
1,001.jpg,0.0
2,002.jpg,4.0
3,003.jpg,8.0
4,004.jpg,12.0
5,005.jpg,16.0
6,006.jpg,20.0
7,007.jpg,24.0
8,008.jpg,28.0
9,009.jpg,32.0
10,010.jpg,36.0
11,011.jpg,40.0
12,012.jpg,44.0
13,013.jpg,48.0
14,014.jpg,52.0
15,015.jpg,56.0
16,016.jpg,60.0
17,017.jpg,64.0
18,018.jpg,68.0
19,019.jpg,72.0
20,020.jpg,76.0
21,021.jpg,80.0
22,022.jpg,84.0
23,023.jpg,88.0
24,024.jpg,92.0
25,025.jpg,96.0
26,026.jpg,100.0
27,027.jpg,104.0
28,028.jpg,108.0
29,029.jpg,112.0
30,030.jpg,116.0
31,031.jpg,120.0
32,032.jpg,124.0
33,033.jpg,128.0
34,034.jpg,132.0
35,035.jpg,136.0
36,036.jpg,140.0
37,037.jpg,144.0
38,038.jpg,148.0
39,039.jpg,152.0
40,040.jpg,156.0
41,041.jpg,160.0
42,042.jpg,164.0
43,043.jpg,168.0
44,044.jpg,172.0
45,045.jpg,176.0
46,046.jpg,180.0
47,047.jpg,184.0
48,048.jpg,188.0
49,049.jpg,192.0
50,050.jpg,196.0
51,051.jpg,200.0
52,052.jpg,204.0
53,053.jpg,208.0
54,054.jpg,212.0
55,055.jpg,216.0
56,056.jpg,220.0
57,057.jpg,224.0
58,058.jpg,228.0
59,059.jpg,232.0
60,060.jpg,236.0
61,061.jpg,240.0
62,062.jpg,244.0
63,063.jpg,248.0
64,064.jpg,252.0
65,065.jpg,256.0
66,066.jpg,260.0
67,067.jpg,264.0
68,068.jpg,268.0
69,069.jpg,272.0
70,070.jpg,276.0
71,071.jpg,280.0
72,072.jpg,284.0
73,073.jpg,288.0
74,074.jpg,292.0
75,075.jpg,296.0
76,076.jpg,300.0
77,077.jpg,304.0
78,078.jpg,308.0
79,079.jpg,312.0
80,080.jpg,316.0
81,081.jpg,320.0
82,082.jpg,324.0
83,083.jpg,328.0
84,084.jpg,332.0
85,085.jpg,336.0
86,086.jpg,340.0
87,087.jpg,344.0
88,088.jpg,348.0
89,089.jpg,352.0
90,090.jpg,356.0
91,091.jpg,360.0
92,092.jpg,364.0
93,093.jpg,368.0
94,094.jpg,372.0
95,095.jpg,376.0
96,096.jpg,380.0
97,097.jpg,384.0
98,098.jpg,388.0
99,099.jpg,392.0
100,100.jpg,396.0
101,101.jpg,400.0
102,102.jpg,404.0
103,103.jpg,408.0
104,104.jpg,412.0
105,105.jpg,416.0
106,106.jpg,420.0
107,107.jpg,424.0
108,108.jpg,428.0
109,109.jpg,432.0
110,110.jpg,436.0
111,111.jpg,440.0
112,112.jpg,444.0
113,113.jpg,448.0
114,114.jpg,452.0
115,115.jpg,456.0
116,116.jpg,460.0
117,117.jpg,464.0
118,118.jpg,468.0
119,119.jpg,472.0
120,120.jpg,476.0
121,121.jpg,480.0
122,122.jpg,484.0
123,123.jpg,488.0
124,124.jpg,492.0
125,125.jpg,496.0
126,126.jpg,500.0
127,127.jpg,504.0
128,128.jpg,508.0
129,129.jpg,512.0
130,130.jpg,516.0
131,131.jpg,520.0
132,132.jpg,524.0
133,133.jpg,528.0
134,134.jpg,532.0
135,135.jpg,536.0
136,136.jpg,540.0
137,137.jpg,544.0
138,138.jpg,548.0
139,139.jpg,552.0
140,140.jpg,556.0
141,141.jpg,560.0
142,142.jpg,564.0
143,143.jpg,568.0
144,144.jpg,572.0
145,145.jpg,576.0
146,146.jpg,580.0
147,147.jpg,584.0
148,148.jpg,588.0
149,149.jpg,592.0
150,150.jpg,596.0
151,151.jpg,600.0
152,152.jpg,604.0
153,153.jpg,608.0
154,154.jpg,612.0
155,155.jpg,616.0
156,156.jpg,620.0
157,157.jpg,624.0
158,158.jpg,628.0
159,159.jpg,632.0
160,160.jpg,636.0
161,161.jpg,640.0
162,162.jpg,644.0
163,163.jpg,648.0
164,164.jpg,652.0
165,165.jpg,656.0
166,166.jpg,660.0
167,167.jpg,664.0
168,168.jpg,668.0
169,169.jpg,672.0
170,170.jpg,676.0
171,171.jpg,680.0
172,172.jpg,684.0
173,173.jpg,688.0
174,174.jpg,692.0
175,175.jpg,696.0
176,176.jpg,700.0
177,177.jpg,704.0
178,178.jpg,708.0
179,179.jpg,712.0
180,180.jpg,716.0
181,181.jpg,720.0
182,182.jpg,724.0
183,183.jpg,728.0
184,184.jpg,732.0
185,185.jpg,736.0
186,186.jpg,740.0
187,187.jpg,744.0
188,188.jpg,748.0
189,189.jpg,752.0
190,190.jpg,756.0
191,191.jpg,760.0
192,192.jpg,764.0
193,193.jpg,768.0
194,194.jpg,772.0
195,195.jpg,776.0
196,196.jpg,780.0
197,197.jpg,784.0
198,198.jpg,788.0
199,199.jpg,792.0
200,200.jpg,796.0
201,201.jpg,800.0
202,202.jpg,804.0
203,203.jpg,808.0
204,204.jpg,812.0
205,205.jpg,816.0
206,206.jpg,820.0
207,207.jpg,824.0
208,208.jpg,828.0
209,209.jpg,832.0
210,210.jpg,836.0
211,211.jpg,840.0
212,212.jpg,844.0
213,213.jpg,848.0
214,214.jpg,852.0
215,215.jpg,856.0
216,216.jpg,860.0
217,217.jpg,864.0
218,218.jpg,868.0
219,219.jpg,872.0
220,220.jpg,876.0
221,221.jpg,880.0
222,222.jpg,884.0
223,223.jpg,888.0
224,224.jpg,892.0
225,225.jpg,896.0
226,226.jpg,900.0
227,227.jpg,904.0
228,228.jpg,908.0
229,229.jpg,912.0
230,230.jpg,916.0
231,231.jpg,920.0
232,232.jpg,924.0
233,233.jpg,928.0
234,234.jpg,932.0
235,235.jpg,936.0
236,236.jpg,940.0
237,237.jpg,944.0
238,238.jpg,948.0
239,239.jpg,952.0
240,240.jpg,956.0
241,241.jpg,960.0
242,242.jpg,964.0
243,243.jpg,968.0
244,244.jpg,972.0
245,245.jpg,976.0
246,246.jpg,980.0
247,247.jpg,984.0
248,248.jpg,988.0
249,249.jpg,992.0
250,250.jpg,996.0
251,251.jpg,1000.0
252,252.jpg,1004.0
253,253.jpg,1008.0
254,254.jpg,1012.0
255,255.jpg,1016.0
256,256.jpg,1020.0
257,257.jpg,1024.0
258,258.jpg,1028.0
259,259.jpg,1032.0
260,260.jpg,1036.0
261,261.jpg,1040.0
262,262.jpg,1044.0
263,263.jpg,1048.0
264,264.jpg,1052.0
265,265.jpg,1056.0
266,266.jpg,1060.0
267,267.jpg,1064.0
268,268.jpg,1068.0
269,269.jpg,1072.0
270,270.jpg,1076.0
271,271.jpg,1080.0
272,272.jpg,1084.0
273,273.jpg,1088.0
274,274.jpg,1092.0
275,275.jpg,1096.0
276,276.jpg,1100.0
277,277.jpg,1104.0
278,278.jpg,1108.0
279,279.jpg,1112.0
280,280.jpg,1116.0
281,281.jpg,1120.0
282,282.jpg,1124.0
283,283.jpg,1128.0
284,284.jpg,1132.0
285,285.jpg,1136.0
286,286.jpg,1140.0
287,287.jpg,1144.0
288,288.jpg,1148.0
289,289.jpg,1152.0
290,290.jpg,1156.0
291,291.jpg,1160.0
292,292.jpg,1164.0
293,293.jpg,1168.0
294,294.jpg,1172.0
295,295.jpg,1176.0
296,296.jpg,1180.0
297,297.jpg,1184.0
298,298.jpg,1188.0
299,299.jpg,1192.0
300,300.jpg,1196.0
301,301.jpg,1200.0
302,302.jpg,1204.0
303,303.jpg,1208.0
304,304.jpg,1212.0
305,305.jpg,1216.0
306,306.jpg,1220.0
307,307.jpg,1224.0



================================================
FILE: data/csv_metadata/video_02.csv
================================================
frame_idx,timestamp
0,0.0
1,0.2
2,0.4
3,0.6
4,0.8
5,1.0
6,1.2
7,1.4
8,1.6
9,1.8
10,2.0
11,2.2
12,2.4
13,2.6
14,2.8
15,3.0
16,3.2
17,3.4
18,3.6
19,3.8
20,4.0
21,4.2
22,4.4
23,4.6
24,4.8
25,5.0
26,5.2
27,5.4
28,5.6
29,5.8
30,6.0
31,6.2
32,6.4
33,6.6
34,6.8
35,7.0
36,7.2
37,7.4
38,7.6
39,7.8
40,8.0
41,8.2
42,8.4
43,8.6
44,8.8
45,9.0
46,9.2
47,9.4
48,9.6
49,9.8
50,10.0
51,10.2
52,10.4
53,10.6
54,10.8
55,11.0
56,11.2
57,11.4
58,11.6
59,11.8
60,12.0
61,12.2
62,12.4
63,12.6
64,12.8
65,13.0
66,13.2
67,13.4
68,13.6
69,13.8
70,14.0
71,14.2
72,14.4
73,14.6
74,14.8
75,15.0
76,15.2
77,15.4
78,15.6
79,15.8
80,16.0
81,16.2
82,16.4
83,16.6
84,16.8
85,17.0
86,17.2
87,17.4
88,17.6
89,17.8
90,18.0
91,18.2
92,18.4
93,18.6
94,18.8
95,19.0
96,19.2
97,19.4
98,19.6
99,19.8
100,20.0
101,20.2
102,20.4
103,20.6
104,20.8
105,21.0
106,21.2
107,21.4
108,21.6
109,21.8
110,22.0
111,22.2
112,22.4
113,22.6
114,22.8
115,23.0
116,23.2
117,23.4
118,23.6
119,23.8
120,24.0
121,24.2
122,24.4
123,24.6
124,24.8
125,25.0
126,25.2
127,25.4
128,25.6
129,25.8
130,26.0
131,26.2
132,26.4
133,26.6
134,26.8
135,27.0
136,27.2
137,27.4
138,27.6
139,27.8
140,28.0
141,28.2
142,28.4
143,28.6
144,28.8
145,29.0
146,29.2
147,29.4
148,29.6
149,29.8
150,30.0
151,30.2
152,30.4
153,30.6
154,30.8
155,31.0
156,31.2
157,31.4
158,31.6
159,31.8
160,32.0
161,32.2
162,32.4
163,32.6
164,32.8
165,33.0
166,33.2
167,33.4
168,33.6
169,33.8
170,34.0
171,34.2
172,34.4
173,34.6
174,34.8
175,35.0
176,35.2
177,35.4
178,35.6
179,35.8
180,36.0
181,36.2
182,36.4
183,36.6
184,36.8
185,37.0
186,37.2
187,37.4
188,37.6
189,37.8
190,38.0
191,38.2
192,38.4
193,38.6
194,38.8
195,39.0
196,39.2
197,39.4
198,39.6
199,39.8



================================================
FILE: data/csv_metadata/video_03.csv
================================================
frame_idx,timestamp
0,0.0
1,0.2
2,0.4
3,0.6
4,0.8
5,1.0
6,1.2
7,1.4
8,1.6
9,1.8
10,2.0
11,2.2
12,2.4
13,2.6
14,2.8
15,3.0
16,3.2
17,3.4
18,3.6
19,3.8
20,4.0
21,4.2
22,4.4
23,4.6
24,4.8
25,5.0
26,5.2
27,5.4
28,5.6
29,5.8
30,6.0
31,6.2
32,6.4
33,6.6
34,6.8
35,7.0
36,7.2
37,7.4
38,7.6
39,7.8
40,8.0
41,8.2
42,8.4
43,8.6
44,8.8
45,9.0
46,9.2
47,9.4
48,9.6
49,9.8
50,10.0
51,10.2
52,10.4
53,10.6
54,10.8
55,11.0
56,11.2
57,11.4
58,11.6
59,11.8
60,12.0
61,12.2
62,12.4
63,12.6
64,12.8
65,13.0
66,13.2
67,13.4
68,13.6
69,13.8
70,14.0
71,14.2
72,14.4
73,14.6
74,14.8
75,15.0
76,15.2
77,15.4
78,15.6
79,15.8
80,16.0
81,16.2
82,16.4
83,16.6
84,16.8
85,17.0
86,17.2
87,17.4
88,17.6
89,17.8
90,18.0
91,18.2
92,18.4
93,18.6
94,18.8
95,19.0
96,19.2
97,19.4
98,19.6
99,19.8
100,20.0
101,20.2
102,20.4
103,20.6
104,20.8
105,21.0
106,21.2
107,21.4
108,21.6
109,21.8
110,22.0
111,22.2
112,22.4
113,22.6
114,22.8
115,23.0
116,23.2
117,23.4
118,23.6
119,23.8


