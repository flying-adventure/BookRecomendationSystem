import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import random
import os

DATA_DIR = "./data"

# ===============================================
# 1. 데이터 전처리 + ISBN 대표값 매핑 + BERT 텍스트 생성
# ===============================================
def load_and_preprocess():

    # -------------------------
    # Books.csv 불러오기
    # -------------------------
    books = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), low_memory=False)

    # 텍스트 생성 (제목 + 저자)
    books["text"] = (
        books["Book-Title"].fillna("") + " [SEP] " +
        books["Book-Author"].fillna("")
    )

    print("\n=== 중복 ISBN 처리 시작 ===")

    # -------------------------
    # (1) 대표 ISBN 만들기
    # Book-Title + Book-Author 기준으로 대표 책 선택
    # -------------------------
    canonical_books = books.groupby(["Book-Title", "Book-Author"]).first().reset_index()

    # 대표 ISBN 목록
    canonical_isbns = canonical_books["ISBN"].tolist()

    # -------------------------
    # (2) ISBN → 대표 ISBN 매핑 생성
    # -------------------------
    isbn_map = {}

    for _, row in books.iterrows():
        title = row["Book-Title"]
        author = row["Book-Author"]
        isbn = row["ISBN"]

        # 대표책의 ISBN 찾기
        rep_isbn = canonical_books[
            (canonical_books["Book-Title"] == title) &
            (canonical_books["Book-Author"] == author)
        ]["ISBN"].values[0]

        isbn_map[isbn] = rep_isbn

    print("ISBN 매핑 개수:", len(isbn_map))

    print("=== 중복 처리 완료 ===\n")

    # 대표 책 테이블만 사용
    books = canonical_books.reset_index(drop=True)

    return books, isbn_map


# ===============================================
# 2. Ratings.csv에 ISBN 매핑 적용
# ===============================================
def load_ratings_with_mapping(isbn_map):
    ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), low_memory=False)

    before = len(ratings)
    ratings["ISBN"] = ratings["ISBN"].map(isbn_map).fillna(ratings["ISBN"])
    after = len(ratings)

    print(f"Ratings ISBN 매핑 적용: {before} → {after}")
    return ratings


# ===============================================
# 3. BERT Sentence Embedding 생성
# ===============================================
def build_bert_embeddings(texts):
    print("=== BERT Sentence Embedding 생성 시작 ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    print("=== BERT Sentence Embedding 생성 완료 ===")
    return model, embeddings


# ===============================================
# 4. 사용자 취향 벡터 생성
# ===============================================
def build_user_vector(user_id, ratings, books, book_emb):
    user_ratings = ratings[ratings["User-ID"] == user_id]

    if len(user_ratings) < 2:
        return None, None

    vectors = []
    weights = []

    for _, row in user_ratings.iterrows():
        isbn = row["ISBN"]
        rating = row["Book-Rating"]

        idx = books.index[books["ISBN"] == isbn]
        if len(idx) == 0:
            continue
        idx = idx[0]

        vectors.append(book_emb[idx])
        weights.append(rating)

    if len(vectors) == 0:
        return None, None

    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-9)
    user_vec = sum(v * w for v, w in zip(vectors, weights))

    return user_vec, user_ratings


# ===============================================
# 5. 책 추천
# ===============================================
def recommend_books(user_vec, book_emb, books, top_k=10):
    scores = util.cos_sim(user_vec, book_emb)[0]
    top_idx = scores.topk(top_k).indices.tolist()
    top_scores = scores[top_idx].tolist()
    return top_idx, top_scores


# ===============================================
# 6. 설명 생성
# ===============================================
def explain_recommendation(user_vec, book_vec):
    sim = util.cos_sim(user_vec, book_vec).item()
    return f"cosine={sim:.4f}"


# ===============================================
# 7. LOO 평가 (상세)
# ===============================================
def evaluate_one_user(user_id, books, ratings, book_emb, model):
    user_ratings = ratings[ratings["User-ID"] == user_id]
    if len(user_ratings) < 2:
        return None

    hidden = user_ratings.sample(1).iloc[0]
    hidden_isbn = hidden["ISBN"]
    hidden_title = books.loc[books["ISBN"] == hidden_isbn, "Book-Title"].values[0]

    ratings_copy = ratings[
        ~((ratings["User-ID"] == user_id) & (ratings["ISBN"] == hidden_isbn))
    ]

    user_vec, _ = build_user_vector(user_id, ratings_copy, books, book_emb)
    if user_vec is None:
        return None

    top_idx, top_scores = recommend_books(user_vec, book_emb, books, top_k=10)

    print(f"\n========= 상세 평가: User {user_id} =========\n")
    print(f"[숨긴 책] {hidden_title} | ISBN={hidden_isbn}\n")
    print("[추천 TOP 10 목록]")

    found_rank = None

    for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
        rec_title = books.iloc[idx]["Book-Title"]
        rec_isbn = books.iloc[idx]["ISBN"]

        explanation = explain_recommendation(user_vec, book_emb[idx])

        print(f"{rank:2d}. {rec_title} | score={score:.4f}")
        print(f"    추천 이유: {explanation}")

        if rec_isbn == hidden_isbn:
            found_rank = rank

    if found_rank is None:
        print("\n✘ 숨긴 책이 TOP 10에 없음 → MISS")
    else:
        print(f"\n✔ 숨긴 책이 {found_rank}위에 존재 → HIT")

    print("===============================================\n")
    return found_rank is not None, found_rank

# ===============================================
# 8. 100명 HR@5 평가
# ===============================================
def evaluate_one_user_simple(user_id, books, ratings, book_emb):
    user_ratings = ratings[ratings["User-ID"] == user_id]
    if len(user_ratings) < 2:
        return None

    hidden = user_ratings.sample(1).iloc[0]
    hidden_isbn = hidden["ISBN"]

    ratings_copy = ratings[
        ~((ratings["User-ID"] == user_id) &
          (ratings["ISBN"] == hidden_isbn))
    ]

    user_vec, _ = build_user_vector(user_id, ratings_copy, books, book_emb)
    if user_vec is None:
        return None

    top_idx, _ = recommend_books(user_vec, book_emb, books, top_k=5)
    top_isbns = [books.iloc[i]["ISBN"] for i in top_idx]

    return hidden_isbn in top_isbns


def evaluate_many_users(books, ratings, book_emb, model, sample_size=100):
    users = ratings["User-ID"].unique()
    sampled = np.random.choice(users, size=sample_size, replace=False)

    hit = 0
    valid = 0

    for user_id in sampled:
        result = evaluate_one_user_simple(user_id, books, ratings, book_emb)
        if result is None:
            continue
        valid += 1
        hit += int(result)

    hr5 = hit / (valid + 1e-9)

    print("\n===== HR@5 평가 결과 (100명) =====")
    print(f"평가 사용자 수: {valid}")
    print(f"HIT: {hit}")
    print(f"HR@5: {hr5:.4f}")
    print("==============================\n")


# ===============================================
# MAIN
# ===============================================
if __name__ == "__main__":

    print("=== 1. 전처리 시작 ===")
    books, isbn_map = load_and_preprocess()
    ratings = load_ratings_with_mapping(isbn_map)
    print("=== 1. 전처리 완료 ===\n")

    print("=== 2. BERT Embedding 생성 ===")
    model, book_emb = build_bert_embeddings(books["text"].tolist())

    # 상세 평가 1명
    random_user = ratings["User-ID"].sample(1).iloc[0]
    evaluate_one_user(random_user, books, ratings, book_emb, model)

    # 100명 HR@5 평가
    evaluate_many_users(books, ratings, book_emb, model)
