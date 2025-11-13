import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz, csr_matrix
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
import time
import os

DATA_DIR = './data'

def get_book_details():
    try:
        books_df = pd.read_csv(os.path.join(DATA_DIR, 'Books.csv'), low_memory=False)
        books_df = books_df[['ISBN', 'Book-Title', 'Book-Author']].set_index('ISBN')
        books_df = books_df.fillna('N/A')
        return books_df
    except FileNotFoundError:
        return pd.DataFrame()
    
def load_data():
    print(f"--- 1. 데이터 로드 시작 (경로: {os.path.join(os.getcwd(), DATA_DIR)}) ---")
    try:
        interactions = load_npz(os.path.join(DATA_DIR, 'interactions.npz'))
        user_features = load_npz(os.path.join(DATA_DIR, 'user_features.npz'))
        item_features = load_npz(os.path.join(DATA_DIR, 'item_features.npz'))
        
        user_features.data[~np.isfinite(user_features.data)] = 0
        item_features.data[~np.isfinite(item_features.data)] = 0

        # ID <-> Mapped ID 변환을 위한 인코더 로드
        with open(os.path.join(DATA_DIR, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)

        books_df = get_book_details()
            
        print(f"Interactions: {interactions.shape}")
        print(f"User: {user_features.shape}")
        print(f"Item: {item_features.shape}")
        print("--- 1. 데이터 로드 완료 ---")
        
        # 학습을 위해 Interactions 행렬을 반환
        return interactions, user_features, item_features, encoders, books_df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"현재 실행 경로: {os.getcwd()}")
        print(f"찾으려 한 경로: {os.path.join(os.getcwd(), DATA_DIR)}")
        print("npz 파일 확인")
        return None, None, None, None, None

def split_data(interactions):
    print("\n--- 2. 데이터 분리 시작 ---")
    # Interactions 행렬 분리
    train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)
    
    # LightFM이 더 빠르게 처리할 수 있는 COOrdinate 형식으로 변환
    train_interactions = train.tocoo()
    test_interactions = test.tocoo()
    
    print(f"훈련 데이터: {train_interactions.nnz}개 상호작용")
    print(f"테스트 데이터: {test_interactions.nnz}개 상호작용")
    print("--- 2. 데이터 분리 완료 ---")
    return train_interactions, test_interactions

def train_model(train_interactions, user_features, item_features):
    print("\n--- 3. 모델 학습 시작 ---")
    start_time = time.time()
    
    # 모델 초기화
    model = LightFM(
        loss='warp',
        no_components=40, 
        learning_rate=0.01,
        user_alpha=1e-3, # L2 정규화
        item_alpha=1e-3, # L2 정규화
        random_state=42
    )
    
    # 모델 학습
    model.fit(
        train_interactions,
        user_features=user_features,
        item_features=item_features,
        sample_weight=None,
        epochs=30,
        num_threads=4,
        verbose=True  # 학습 진행 상황 출력
    )
    
    end_time = time.time()
    print(f"--- 3. 모델 학습 완료 (소요 시간: {end_time - start_time:.2f}초) ---")
    return model

def evaluate_model(model, train_interactions, test_interactions, user_features, item_features):
    print("\n--- 4. 모델 평가 시작 (AUC) ---")
    
    sample_size = 5000
    num_users = train_interactions.shape[0]

    # AUC 평가를 위해 5000명의 유저 인덱스를 무작위로 선택
    sample_users_indices = np.random.choice(num_users, size=sample_size, replace=False)
    
    # interactions을 CSR 형식으로 변환하여 행 필터링을 준비
    train_csr = train_interactions.tocsr()
    test_csr = test_interactions.tocsr()
    
    sampled_train_interactions = train_csr[sample_users_indices, :]
    sampled_test_interactions = test_csr[sample_users_indices, :]
    
    sampled_user_features = user_features[sample_users_indices, :] 

    print(f"디버그: {sample_size}명의 유저에 대해 평가 중...")

    # 훈련 데이터셋에 대한 AUC
    train_auc = auc_score(
        model,
        sampled_train_interactions,
        user_features=sampled_user_features,
        item_features=item_features,
        num_threads=4,
    ).mean()
    
    # 테스트 데이터셋에 대한 AUC
    test_auc = auc_score(
        model,
        sampled_test_interactions,
        train_interactions=sampled_train_interactions,
        user_features=sampled_user_features,
        item_features=item_features,
        num_threads=4,
    ).mean()
    
    print("\n--- 5. 최종 평가 결과 ---")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")
    print("------------------------")

def predict_recommendations(model, user_features, item_features, encoders, books_df, user_id_original, num_recommendations=5):
    user_encoder = encoders['user_encoder']
    item_encoder = encoders['item_encoder']

    user_id_str = str(user_id)

    try:
        user_internal_id = encoders['user_encoder'].transform([user_id_str])[0]
        
    except ValueError:
        print(f"ERROR: User id {user_id_original} does not exist in the learned dataset.")
        return

    # 추천 점수 예측
    num_items = item_features.shape[0]
    
    user_ids_to_predict = np.repeat(user_internal_id, num_items)
    item_ids_to_predict = np.arange(num_items)
    
    scores = model.predict(
        user_ids=user_ids_to_predict,
        item_ids=item_ids_to_predict,
        user_features=user_features,
        item_features=item_features,
        num_threads=4
    )
    
    # 점수를 내림차순으로 정렬하여 상위 N개 아이템의 index 확인
    top_indices = np.argsort(-scores)[:num_recommendations]
    
    # LightFM 내부 아이템 ID를 원래의 ISBN ID로 역변환
    top_original_ids = item_encoder.inverse_transform(top_indices)
    
    print(f"Top {num_recommendations} books recommended to user ID: {user_id_original}")
    
    # 추천 결과를 Pandas를 사용해 책 제목과 함께 출력
    for rank, (isbn, index) in enumerate(zip(top_original_ids, top_indices)):
        try:
            # ISBN을 사용하여 Books DataFrame에서 책 제목/저자 조회
            book_info = books_df.loc[isbn]
            title = book_info['Book-Title']
            author = book_info['Book-Author']
        except KeyError:
            title = "No title information found"
            author = "N/A"
            
        print(f"{rank+1}. {title} (By: {author})")
        print(f"   ISBN: {isbn}, Predicted score: {scores[index]:.4f}")
        
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    interactions, user_features, item_features, encoders, books_df = load_data() 
    
    if interactions is not None:
        train_data, test_data = split_data(interactions)
        model = train_model(train_data, user_features, item_features)
        evaluate_model(model, train_data, test_data, user_features, item_features)

        # 사용자 ID 입력 받기
        print("Book Recomendation System")
        while True:
            try:
                # 사용자 ID를 입력받도록 요청 (숫자만 허용)
                user_id_input = input("Enter the user ID (Exit: q):")
                user_id_str = str(user_id_input)

                if user_id_input.lower() == 'q':
                    print("--- Exit Program ---")
                    break

                if not user_id_input.isdigit():
                    print("Invalid input, User ID must be a number.")
                    continue

                int(user_id_str)
                user_id = user_id_input 

                # 예측 실행
                predict_recommendations(model, user_features, item_features, encoders, books_df, user_id, num_recommendations=5)
            
            except EOFError:
                print("\nEnd input")
                user_id = None
                break
            except KeyboardInterrupt:
                print("\nStop running.")
                user_id = None
                break