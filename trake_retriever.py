import os
import numpy as np
import pandas as pd
import torch
import open_clip
from pymilvus import MilvusClient

# Cấu hình mặc định
MILVUS_URI = "./milvus_demo.db"
COLLECTION_NAME = "clip_keyframes"
CSV_DIR = "data/csv_metadata"
FEATURE_DIR = "data/clip_features"


def normalize(v: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 cho mảng vector."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm == 0, v, v / norm)


class TrakeEngine:
    def __init__(
        self,
        milvus_uri: str = MILVUS_URI,
        collection_name: str = COLLECTION_NAME,
        csv_dir: str = CSV_DIR,
        feature_dir: str = FEATURE_DIR,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        clip_model=None,
        tokenizer=None,
        device: str = None,
    ):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.csv_dir = csv_dir
        self.feature_dir = feature_dir

        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            pass

        # Quản lý thiết bị tính toán
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Khởi tạo CLIP Text Encoder
        if clip_model is not None and tokenizer is not None:
            self.clip_model = clip_model
            self.tokenizer = tokenizer
        else:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained
            )
            self.clip_model = self.clip_model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Bộ nhớ đệm RAM
        self._text_embed_cache: dict[str, np.ndarray] = {}
        self._feature_cache: dict[str, tuple[np.ndarray, list[int]]] = {}

    def _encode_text(self, text: str) -> np.ndarray:
        """Chuyển văn bản thành vector CLIP (512 chiều) chuẩn hóa L2 và cache kết quả."""
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
        """Truy vấn Milvus theo từng event con và vote điểm để chọn Top-K video tiềm năng."""
        video_scores: dict[str, float] = {}

        for ev in structured_query.events:
            query_vec = self._encode_text(ev.description)
            search_res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vec.tolist()],
                anns_field="embedding",
                search_params={"metric_type": "IP", "params": {}},
                limit=50,
                output_fields=["video_id"],
            )
            if search_res and len(search_res) > 0:
                for hit in search_res[0]:
                    vid = hit["entity"]["video_id"]
                    video_scores[vid] = video_scores.get(vid, 0.0) + hit["distance"]

        ranked = sorted(video_scores.items(), key=lambda x: -x[1])
        return [vid for vid, _ in ranked[:top_k]]

    def _load_video_features(self, video_id: str) -> tuple[np.ndarray | None, list[int] | None]:
        """Đọc và cache đặc trưng vector (.npy) và danh sách frame_idx (.csv) của video."""
        if video_id in self._feature_cache:
            return self._feature_cache[video_id]

        csv_file = os.path.join(self.csv_dir, f"{video_id}.csv")
        npy_file = os.path.join(self.feature_dir, f"{video_id}.npy")

        if not os.path.exists(csv_file) or not os.path.exists(npy_file):
            return None, None

        df = pd.read_csv(csv_file)
        if "frame_idx" in df.columns:
            frame_ids = df["frame_idx"].tolist()
        elif "frame_id" in df.columns:
            frame_ids = df["frame_id"].tolist()
        else:
            frame_ids = list(range(1, len(df) + 1))

        features = np.load(npy_file).astype(np.float32)
        features = normalize(features)

        self._feature_cache[video_id] = (features, frame_ids)
        return features, frame_ids

    def solve_trake(
        self,
        structured_query,
        top_k_videos: int = 20,
        min_gap: int = 1,
    ) -> list[tuple[str, list[int], float]]:
        """Trả về danh sách các video ứng viên đã căn chỉnh frame, sắp xếp theo điểm giảm dần."""
        events = structured_query.events
        N = len(events)
        candidate_vids = self.retrieve_candidate_videos(
            structured_query, top_k=top_k_videos
        )

        if not candidate_vids:
            return []

        event_embeddings = np.stack([self._encode_text(ev.description) for ev in events])
        results = []

        for vid in candidate_vids:
            features, frame_ids = self._load_video_features(vid)
            if features is None or len(features) < N:
                continue

            T = len(features)
            sim_matrix = np.dot(event_embeddings, features.T)

            dp = np.full((N, T), -np.inf, dtype=np.float32)
            backtrack = np.full((N, T), -1, dtype=np.int32)

            dp[0, :] = sim_matrix[0, :]

            for i in range(1, N):
                prefix_max = np.full(T, -np.inf, dtype=np.float32)
                prefix_arg = np.full(T, -1, dtype=np.int32)
                running_best = -np.inf
                running_best_idx = -1

                for t in range(T):
                    if dp[i - 1, t] > running_best:
                        running_best = dp[i - 1, t]
                        running_best_idx = t
                    prefix_max[t] = running_best
                    prefix_arg[t] = running_best_idx

                for t in range(i * min_gap, T):
                    src = t - min_gap
                    if src < 0:
                        continue
                    if prefix_max[src] != -np.inf:
                        dp[i, t] = prefix_max[src] + sim_matrix[i, t]
                        backtrack[i, t] = prefix_arg[src]

            last_t = int(np.argmax(dp[N - 1, :]))
            current_score = float(dp[N - 1, last_t])

            if current_score != -np.inf and last_t != -1:
                path = [last_t]
                curr = last_t
                valid_path = True
                for i in range(N - 1, 0, -1):
                    curr = backtrack[i, curr]
                    if curr == -1:
                        valid_path = False
                        break
                    path.append(curr)

                if valid_path:
                    path.reverse()
                    aligned_frames = [int(frame_ids[idx]) for idx in path]
                    # Làm sạch tên video (loại bỏ đuôi .mp4 nếu có)
                    clean_vid = vid.replace(".mp4", "")
                    results.append((clean_vid, aligned_frames, current_score))

        # Sắp xếp theo tổng điểm tương đồng giảm dần
        results.sort(key=lambda x: -x[2])
        return results