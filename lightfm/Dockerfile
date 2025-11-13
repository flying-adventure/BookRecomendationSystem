# 1. 베이스 이미지를 설정합니다.
FROM python:3.10-slim-bookworm

# 2. 시스템 의존성 설치 (C++ 컴파일러 등)
RUN apt-get update && apt-get install -y build-essential -y

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. Python 라이브러리 설치 (필수 패키지)
RUN pip install lightfm pandas scikit-learn numpy scipy

# 5. 호스트(PC)에 있는 프로젝트 파일 및 데이터를 컨테이너로 복사합니다.
# 현재 파일 위치: src/recommender_system.py -> 컨테이너의 /app/recommender_system.py로 복사
COPY src /app/src
COPY data /app/data

# 6. 컨테이너가 시작될 때 recommender_system.py 스크립트를 실행합니다.
CMD ["python3", "src/recommender_system.py"]