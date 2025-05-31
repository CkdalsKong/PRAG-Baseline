import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 데이터 로드
eval_df = pd.read_csv('/data/my_PRAG/baseline/output_wdoc/evaluation_report.csv')
gen_df = pd.read_csv('/data/my_PRAG/baseline/output_wdoc/generation_report.csv')
idx_df = pd.read_csv('/data/my_PRAG/baseline/output_wdoc/indexing_report.csv')

# 1. 성능 지표 분석
def analyze_performance():
    # 각 방법별 평균 성능 계산
    performance = eval_df.groupby('method').agg({
        'unhelpful': 'mean',
        'inconsistent': 'mean',
        'hallucination_of_preference_violation': 'mean',
        'preference_unaware_violation': 'mean',
        'preference_following_accuracy(%)': 'mean'
    }).round(2)
    
    print("\n=== 각 방법별 평균 성능 ===")
    print(performance)
    
    # 시각화
    plt.figure(figsize=(12, 6))
    performance['preference_following_accuracy(%)'].plot(kind='bar')
    plt.title('Preference Following Accuracy by Method')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_accuracy.png')
    
    # 오류 유형별 시각화
    error_types = ['unhelpful', 'inconsistent', 'hallucination_of_preference_violation', 'preference_unaware_violation']
    plt.figure(figsize=(12, 6))
    performance[error_types].plot(kind='bar', stacked=True)
    plt.title('Error Types Distribution by Method')
    plt.ylabel('Number of Errors')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_types.png')

# 2. 처리 시간 분석
def analyze_processing_time():
    # 각 방법별 평균 처리 시간 계산
    time_metrics = idx_df.groupby('method').agg({
        'cosine_filter_time(s)': 'mean',
        'llm_time(s)': 'mean',
        'summary_time(s)': 'mean',
        'faiss_time(s)': 'mean',
        'total_time(s)': 'mean'
    }).round(2)
    
    print("\n=== 각 방법별 평균 처리 시간 (초) ===")
    print(time_metrics)
    
    # 처리 시간 시각화
    plt.figure(figsize=(12, 6))
    time_metrics[['cosine_filter_time(s)', 'llm_time(s)', 'summary_time(s)', 'faiss_time(s)']].plot(kind='bar', stacked=True)
    plt.title('Processing Time Distribution by Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('processing_time.png')

# 3. 검색 성능 분석
def analyze_retrieval():
    # 각 방법별 평균 검색 시간 계산
    retrieval_time = gen_df.groupby('method')['avg_retrieval_time(s)'].mean().round(4)
    
    print("\n=== 각 방법별 평균 검색 시간 (초) ===")
    print(retrieval_time)
    
    # 검색 시간 시각화
    plt.figure(figsize=(10, 6))
    retrieval_time.plot(kind='bar')
    plt.title('Average Retrieval Time by Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('retrieval_time.png')

# 4. 데이터 처리량 분석
def analyze_data_volume():
    # 각 방법별 데이터 처리량 계산
    data_volume = idx_df.groupby('method').agg({
        'cosine_kept': 'mean',
        'llm_filtered': 'mean',
        'summarized': 'mean',
        'kept': 'mean'
    }).round(2)
    
    print("\n=== 각 방법별 평균 데이터 처리량 ===")
    print(data_volume)
    
    # 데이터 처리량 시각화
    plt.figure(figsize=(12, 6))
    data_volume[['cosine_kept', 'llm_filtered', 'summarized', 'kept']].plot(kind='bar', stacked=True)
    plt.title('Data Processing Volume by Method')
    plt.ylabel('Number of Chunks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data_volume.png')

# 5. 종합 분석 보고서 생성
def generate_summary_report():
    print("\n=== 종합 분석 보고서 ===")
    
    # 1. 성능 순위
    performance_rank = eval_df.groupby('method')['preference_following_accuracy(%)'].mean().sort_values(ascending=False)
    print("\n1. 선호도 준수 정확도 순위:")
    print(performance_rank)
    
    # 2. 처리 시간 순위
    time_rank = idx_df.groupby('method')['total_time(s)'].mean().sort_values()
    print("\n2. 총 처리 시간 순위 (빠른 순):")
    print(time_rank)
    
    # 3. 검색 시간 순위
    retrieval_rank = gen_df.groupby('method')['avg_retrieval_time(s)'].mean().sort_values()
    print("\n3. 평균 검색 시간 순위 (빠른 순):")
    print(retrieval_rank)
    
    # 4. 데이터 처리 효율성
    efficiency = idx_df.groupby('method').agg({
        'kept': 'mean',
        'total_time(s)': 'mean'
    })
    efficiency['chunks_per_second'] = (efficiency['kept'] / efficiency['total_time(s)']).round(2)
    print("\n4. 초당 처리 청크 수:")
    print(efficiency['chunks_per_second'].sort_values(ascending=False))

if __name__ == "__main__":
    # 분석 실행
    analyze_performance()
    analyze_processing_time()
    analyze_retrieval()
    analyze_data_volume()
    generate_summary_report() 