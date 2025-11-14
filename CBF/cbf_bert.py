import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import random
import os

DATA_DIR = "./data"


# ============================================================
# 1. 데이터 전처리 + ISBN 대표 매핑 + 텍스트 생성 + 연도 정규화
# ============================================================
def load_and_preprocess():
    print("=== 1. 데이터 전처리 시작 ===")

    try:
        books = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), low_memory=False)
    except FileNotFoundError:
        print("Books.csv 파일 없음. 경로 확인 필요.")
        return None, None

    # 텍스트 기본 생성 (제목 + 저자)
    books["text"] = (
        books["Book-Title"].fillna("") + " [SEP] " +
        books["Book-Author"].fillna("")
    )

    print("\n=== 중복 ISBN 처리 시작 (Title/Author 기준) ===")

    # 대표 ISBN 만들기
    canonical_books_df = books.groupby(["Book-Title", "Book-Author"]).first().reset_index()

    canonical_books_rep_isbn = canonical_books_df[
        ["Book-Title", "Book-Author", "ISBN"]
    ].rename(columns={"ISBN": "rep_ISBN"})

    books_with_rep_isbn = pd.merge(
        books[["ISBN", "Book-Title", "Book-Author"]],
        canonical_books_rep_isbn,
        on=["Book-Title", "Book-Author"],
        how="left"
    )

    isbn_map = pd.Series(
        books_with_rep_isbn["rep_ISBN"].values,
        index=books_with_rep_isbn["ISBN"]
    ).to_dict()

    print("ISBN 매핑 개수:", len(isbn_map))
    print("=== 중복 처리 완료 ===\n")

    # 대표 책 테이블만 사용
    books = canonical_books_df.reset_index(drop=True)

    # 텍스트 재생성
    books["text"] = (
        books["Book-Title"].fillna("") + " [SEP] " +
        books["Book-Author"].fillna("")
    )

    # ---------------------------------------------------------
    # 출판 연도 정규화 추가
    # ---------------------------------------------------------
    print("=== 출판연도 정규화 ===")

    # 숫자 변환
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"], errors="coerce"
    )

    # 중앙값으로 대체
    year_median = books["Year-Of-Publication"].median()
    books["Year-Of-Publication"] = books["Year-Of-Publication"].fillna(year_median)

    # 정규화
    year_min = books["Year-Of-Publication"].min()
    year_max = books["Year-Of-Publication"].max()

    books["year_norm"] = (
        (books["Year-Of-Publication"] - year_min) /
        (year_max - year_min + 1e-9)
    )

    print("출판연도 정규화 완료\n")

    return books, isbn_map


# ============================================================
# 2. Ratings.csv에 ISBN 매핑 적용
# ============================================================
def load_ratings_with_mapping(isbn_map):
    print("=== 2. Ratings 데이터 로드 및 매핑 적용 ===")

    try:
        ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), low_memory=False)
    except FileNotFoundError:
        print("Ratings.csv 파일 없음")
        return None

    before = len(ratings)
    ratings["ISBN"] = ratings["ISBN"].map(isbn_map).fillna(ratings["ISBN"])
    after = len(ratings)

    print(f"Ratings ISBN 매핑 적용 완료: {before}행 유지\n")
    return ratings


# ============================================================
# 3. BERT 임베딩
# ============================================================
def build_bert_embeddings(texts):
    print("=== 3. BERT Sentence Embedding 생성 (all-MiniLM-L6-v2) ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    print("BERT 임베딩 완료\n")
    return model, embeddings


# ============================================================
# 4. 사용자 벡터 생성 (출판연도 포함 Concatenate)
# ============================================================
def build_user_vector(user_id, ratings, books, final_emb):
    user_ratings = ratings[ratings["User-ID"] == user_id]

    if len(user_ratings) < 2:
        return None, None

    vectors = []
    weights = []

    ebook = pd.Series(books.index.values, index=books["ISBN"]).to_dict()

    for _, row in user_ratings.iterrows():
        isbn = row["ISBN"]
        rating = row["Book-Rating"]

        idx = ebook.get(isbn)
        if idx is None:
            continue

        vectors.append(final_emb[idx])
        weights.append(rating)

    if len(vectors) == 0:
        return None, None

    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-9)

    user_vec = vectors[0] * weights[0]
    for v, w in zip(vectors[1:], weights[1:]):
        user_vec += v * w

    return user_vec, user_ratings


# ============================================================
# 5. 추천
# ============================================================
def recommend_books(user_vec, final_emb, books, top_k=10):
    scores = util.cos_sim(user_vec, final_emb)[0]
    top_results = scores.topk(top_k)

    return top_results.indices.tolist(), top_results.values.tolist()


# ============================================================
# 6. 설명 생성
# ============================================================
def explain_recommendation(user_vec, book_vec):
    sim = util.cos_sim(user_vec, book_vec).item()
    return f"cosine={sim:.4f}"


# ============================================================
# 7. 상세 평가 (LOO)
# ============================================================
def evaluate_one_user(user_id, books, ratings, final_emb, model):
    print(f"\n========= 상세 평가: User {user_id} =========\n")

    user_ratings = ratings[ratings["User-ID"] == user_id]
    if len(user_ratings) < 2:
        print("평가 불가\n")
        return None

    hidden = user_ratings.sample(1).iloc[0]
    hidden_isbn = hidden["ISBN"]

    if hidden_isbn not in books["ISBN"].values:
        print("숨긴 책이 대표 목록에 없음\n")
        return None

    hidden_title = books.loc[books["ISBN"] == hidden_isbn, "Book-Title"].values[0]

    ratings_copy = ratings[
        ~((ratings["User-ID"] == user_id) & (ratings["ISBN"] == hidden_isbn))
    ]

    user_vec, _ = build_user_vector(user_id, ratings_copy, books, final_emb)
    if user_vec is None:
        print("남은 책 부족\n")
        return None

    top_idx, top_scores = recommend_books(user_vec, final_emb, books, top_k=10)

    print(f"[숨긴 책] {hidden_title} | ISBN={hidden_isbn}\n")
    print("[TOP 10 추천]")

    found_rank = None

    for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
        rec_title = books.iloc[idx]["Book-Title"]
        rec_isbn = books.iloc[idx]["ISBN"]
        explanation = explain_recommendation(user_vec, final_emb[idx])

        print(f"{rank}. {rec_title} | score={score:.4f}")
        print(f"   추천 이유: {explanation}")

        if rec_isbn == hidden_isbn:
            found_rank = rank

    if found_rank is None:
        print("\n✘ 숨긴 책이 TOP 10에 없음 → MISS")
    else:
        print(f"\n✔ 숨긴 책이 {found_rank}위 → HIT")

    print("=============================================\n")
    return found_rank is not None, found_rank


# ============================================================
# HR@10 평가
# ============================================================
def evaluate_one_user_simple_hr10(user_id, books, ratings, final_emb):
    user_ratings = ratings[ratings["User-ID"] == user_id]
    if len(user_ratings) < 2:
        return None

    hidden = user_ratings.sample(1).iloc[0]
    hidden_isbn = hidden["ISBN"]

    if hidden_isbn not in books["ISBN"].values:
        return None

    ratings_copy = ratings[
        ~((ratings["User-ID"] == user_id) &
          (ratings["ISBN"] == hidden_isbn))
    ]

    user_vec, _ = build_user_vector(user_id, ratings_copy, books, final_emb)
    if user_vec is None:
        return None

    top_idx, _ = recommend_books(user_vec, final_emb, books, top_k=10)
    top_isbns = [books.iloc[i]["ISBN"] for i in top_idx]

    return hidden_isbn in top_isbns


def evaluate_many_users(books, ratings, final_emb, sample_size=100):
    print("=== HR@10 평가 시작 ===")

    eligible_users = ratings["User-ID"].value_counts()
    eligible_users = eligible_users[eligible_users >= 2].index.tolist()

    if len(eligible_users) < sample_size:
        sample_size = len(eligible_users)

    sampled = np.random.choice(eligible_users, size=sample_size, replace=False)

    hit = 0
    valid = 0

    for user_id in sampled:
        result = evaluate_one_user_simple_hr10(user_id, books, ratings, final_emb)
        if result is None:
            continue
        valid += 1
        hit += int(result)

    hr10 = hit / (valid + 1e-9)

    print("\n===== HR@10 평가 결과 =====")
    print(f"평가 사용자 수: {valid}")
    print(f"HIT: {hit}")
    print(f"HR@10: {hr10:.4f}")
    print("==========================\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    books, isbn_map = load_and_preprocess()
    if books is None:
        exit()

    ratings = load_ratings_with_mapping(isbn_map)
    if ratings is None:
        exit()

    # --- BERT 임베딩 생성 ---
    model, bert_emb = build_bert_embeddings(books["text"].tolist())

    # --- 출판연도 포함 vector concatenate ---
    print("출판연도 벡터 결합 중...")

    year_vec = books["year_norm"].values.reshape(-1, 1)
    final_emb = np.hstack([bert_emb.cpu().numpy(), year_vec])  # shape: (n_books, 385)

    print("결합 완료!\n")

    # 상세 평가 1명
    print("=== 상세 평가 1명 ===")
    eligible_users = ratings["User-ID"].value_counts()
    eligible_users = eligible_users[eligible_users >= 2].index.tolist()

    if eligible_users:
        test_user = random.choice(eligible_users)
        evaluate_one_user(test_user, books, ratings, final_emb, model)

    # HR@10 평가
    evaluate_many_users(books, ratings, final_emb)
