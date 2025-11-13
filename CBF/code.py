import pandas as pd
import numpy as np
import os

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "./data"


# =============================================
#                  전처리
# =============================================
def preprocess_raw_data():
    print("\n=== 1. 전처리 시작 ===")

    books = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), low_memory=False)
    users = pd.read_csv(os.path.join(DATA_DIR, "Users.csv"), low_memory=False)
    ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), low_memory=False)

    books["Book-Title"] = books["Book-Title"].fillna("Unknown Title")
    books["Book-Author"] = books["Book-Author"].fillna("Unknown Author")
    books["Publisher"] = books["Publisher"].fillna("Unknown Publisher")

    ratings = ratings[
        (ratings["Book-Rating"] >= 0) &
        (ratings["Book-Rating"] <= 10)
    ]

    books = books[["ISBN", "Book-Title", "Book-Author"]].set_index("ISBN")

    print("=== 1. 전처리 완료 ===")
    return books, users, ratings


# =============================================
#           ISBN 중복 제거
# =============================================
def deduplicate_books(books):
    print("\n=== ISBN 중복 제거 ===")

    before = len(books)
    df = books.reset_index()
    df = df.drop_duplicates(subset=["Book-Title", "Book-Author"])
    df = df.set_index("ISBN")
    after = len(df)

    print(f"중복 제거: {before} → {after}")
    return df


# =============================================
#              Word2Vec 학습용 토큰화
# =============================================
def tokenize_books(books):
    print("\n=== 2. 책 텍스트 토큰화 ===")

    sentences = []
    for _, row in books.iterrows():
        title = str(row["Book-Title"]).lower().split()
        author = str(row["Book-Author"]).lower().split()
        sentences.append(title + author)
    return sentences


# =============================================
#             Word2Vec 학습
# =============================================
def train_w2v(sentences):
    print("\n=== 3. Word2Vec 학습 시작 ===")
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1  
    )
    print("=== Word2Vec 학습 완료 ===")
    return model


# =============================================
#        책 임베딩 생성 (단어 임베딩 평균)
# =============================================
def build_book_vectors(books, model):
    print("\n=== 4. 책 임베딩 생성 ===")

    vectors = []
    for _, row in books.iterrows():
        words = str(row["Book-Title"]).lower().split() + \
                str(row["Book-Author"]).lower().split()

        valid = [model.wv[w] for w in words if w in model.wv]

        if len(valid) == 0:
            vec = np.zeros(model.vector_size)
        else:
            vec = np.mean(valid, axis=0)

        vectors.append(vec)

    return np.array(vectors)


# =============================================
#            책 벡터 개별 리턴
# =============================================
def get_book_vector(isbn, books, book_vectors):
    isbn_to_idx = {isbn: i for i, isbn in enumerate(books.index)}
    if isbn not in isbn_to_idx:
        return None
    return book_vectors[isbn_to_idx[isbn]]


# =============================================
#              사용자 벡터 생성 (평점 가중치)
# =============================================
def get_user_vector(user_id, ratings, books, book_vectors):
    history = ratings[ratings["User-ID"] == user_id]

    if len(history) == 0:
        return None

    isbn_to_idx = {isbn: i for i, isbn in enumerate(books.index)}

    weighted_sum = np.zeros(book_vectors.shape[1])
    total_weight = 0

    for _, row in history.iterrows():
        isbn = row["ISBN"]
        rating = row["Book-Rating"]

        if isbn not in isbn_to_idx:
            continue

        idx = isbn_to_idx[isbn]
        weighted_sum += book_vectors[idx] * rating
        total_weight += rating

    if total_weight == 0:
        return None

    return weighted_sum / total_weight


# =============================================
#          추천 이유(Explanation) 생성
# =============================================
def explain_recommendation(model, user_vec, book_vec, topn=5):
    user_sim = model.wv.similar_by_vector(user_vec, topn=20)
    book_sim = model.wv.similar_by_vector(book_vec, topn=20)

    user_words = [w for w, _ in user_sim]
    book_words = [w for w, _ in book_sim]

    common = [w for w in user_words if w in book_words]

    if len(common) == 0:
        common = user_words[:3] + book_words[:3]

    return common[:topn]


# =============================================
#                 추천 생성
# =============================================
def recommend_books(user_id, ratings, books, book_vectors, model, top_k=5):
    user_vec = get_user_vector(user_id, ratings, books, book_vectors)
    if user_vec is None:
        return [], None, None

    sim_scores = cosine_similarity([user_vec], book_vectors).flatten()
    top_idx = sim_scores.argsort()[::-1][:top_k]

    rec_isbns = books.iloc[top_idx].index.tolist()
    return rec_isbns, sim_scores[top_idx], user_vec


# =============================================
#                    MAIN
# =============================================
if __name__ == "__main__":
    books, users, ratings = preprocess_raw_data()
    books = deduplicate_books(books)

    sentences = tokenize_books(books)
    w2v = train_w2v(sentences)

    book_vectors = build_book_vectors(books, w2v)

    test_user = ratings["User-ID"].sample(1).iloc[0]

    recs, scores, user_vec = recommend_books(test_user, ratings, books, book_vectors, w2v, top_k=5)

    if recs:
        print(f"\n=== 사용자 {test_user} 추천 ===")

        for isbn, score in zip(recs, scores):
            title = books.loc[isbn]["Book-Title"]
            book_vec = get_book_vector(isbn, books, book_vectors)

            reasons = explain_recommendation(w2v, user_vec, book_vec)

            print(f"- {title} | {score:.4f}")
            print(f"  추천 이유: {', '.join(reasons)}")

    else:
        print(f"\n사용자 {test_user}는 추천 불가 (평점 데이터 부족)")
