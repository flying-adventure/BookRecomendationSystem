import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import random
import os

DATA_DIR = "./data"


# ===============================================
# 1. 데이터 전처리 + BERT 임베딩 준비
# ===============================================
def load_and_preprocess():
    books = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), low_memory=False)

    # 텍스트 생성 (제목 + 저자)
    books["text"] = (
        books["Book-Title"].fillna("") + " [SEP] " +
        books["Book-Author"].fillna("")
    )

    # ISBN 중복 제거
    before = len(books)
    books = books.drop_duplicates(subset=["ISBN"])
    after = len(books)
    print(f"ISBN 중복 제거: {before} → {after}")

    books = books.reset_index(drop=True)
    return books


# ===============================================
# 2. BERT Sentence Embedding 생성
# ===============================================
def build_bert_embeddings(texts):
    print("\n=== BERT Sentence Embedding 생성 시작 ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    print("=== BERT Sentence Embedding 생성 완료 ===")
    return model, embeddings


# ===============================================
# 3. 사용자 취향 벡터 생성
# ===============================================
def build_user_vector(user_id, ratings, books, book_emb):
    user_ratings = ratings[ratings["User-ID"] == user_id]

    if len(user_ratings) < 2:
        return None, None

    # 평점 기준으로 가중 평균
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
# 4. 책 추천
# ===============================================
def recommend_books(user_vec, book_emb, books, top_k=10):
    scores = util.cos_sim(user_vec, book_emb)[0]
    top_idx = scores.topk(top_k).indices.tolist()
    top_scores = scores[top_idx].tolist()

    return top_idx, top_scores


# ===============================================
# 5. 추천 이유 생성 (BERT 기반)
# ===============================================
def explain_recommendation(user_vec, book_vec, model):
    # 가장 유사한 단어/토큰을 뽑아 설명 흉내
    sim = util.cos_sim(user_vec, book_vec).item()
    return f"cosine={sim:.4f}"


# ===============================================
# 6. LOO 평가 한 명 (상세)
# ===============================================
def evaluate_one_user(user_id, books, ratings, book_emb, model):
    user_ratings = ratings[ratings["User-ID"] == user_id]

    if len(user_ratings) < 2:
        return None

    # 숨길 책 선택
    hidden = user_ratings.sample(1).iloc[0]
    hidden_isbn = hidden["ISBN"]
    hidden_title = books.loc[books["ISBN"] == hidden_isbn, "Book-Title"].values[0]

    # 남은 책으로 학습
    remain = user_ratings[user_ratings["ISBN"] != hidden_isbn].copy()

    ratings_copy = ratings.copy()
    ratings_copy = ratings_copy[
        ~((ratings_copy["User-ID"] == user_id) &
          (ratings_copy["ISBN"] == hidden_isbn))
    ]

    user_vec, _ = build_user_vector(user_id, ratings_copy, books, book_emb)
    if user_vec is None:
        return None

    top_idx, top_scores = recommend_books(user_vec, book_emb, books, top_k=10)

    print(f"\n========= 상세 평가 시작: User {user_id} =========\n")
    print(f"[숨긴 책] {hidden_title} | ISBN={hidden_isbn}\n")

    print("[추천 TOP 10 목록]")
    found_rank = None
    hidden_vec = None
    try:
        hidden_vec = book_emb[books.index[books["ISBN"] == hidden_isbn][0]]
    except:
        pass

    for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
        rec_title = books.iloc[idx]["Book-Title"]
        rec_isbn = books.iloc[idx]["ISBN"]

        explanation = ""
        if hidden_vec is not None:
            explanation = explain_recommendation(user_vec, book_emb[idx], model)

        print(f" {rank:2d}. {rec_title} | score={score:.4f}")
        print(f"    추천 이유: {explanation}")

        if rec_isbn == hidden_isbn:
            found_rank = rank

    if found_rank is None:
        print("\n✘ 숨긴 책이 TOP 10에 없음 → MISS")
    else:
        print(f"\n✔ 숨긴 책이 {found_rank}위에 존재 → HIT")

    print("\n=====================================================\n")

    return found_rank is not None, found_rank


# ===============================================
# 7. 100명 HR@5 평가
# ===============================================
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

    print("\n===== 100명 HR@5 평가 결과 =====")
    print(f"총 평가 가능 사용자 수: {valid}")
    print(f"HIT 수: {hit}")
    print(f"HR@5: {hr5:.4f}")
    print("=================================\n")


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


# ===============================================
# MAIN
# ===============================================
if __name__ == "__main__":

    print("=== 1. 전처리 시작 ===")
    books = load_and_preprocess()
    ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), low_memory=False)
    print("=== 1. 전처리 완료 ===\n")

    print("=== 2. BERT 임베딩 생성 ===")
    model, book_emb = build_bert_embeddings(books["text"].tolist())

    # 상세 평가 1명
    random_user = ratings["User-ID"].sample(1).iloc[0]
    evaluate_one_user(random_user, books, ratings, book_emb, model)

    # 100명 HR@5 평가
    evaluate_many_users(books, ratings, book_emb, model)
