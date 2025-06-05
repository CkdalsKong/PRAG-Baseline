import pandas as pd
import matplotlib.pyplot as plt
import warnings

# 한글 폰트 설정 (환경에 따라 다르게 설정하세요)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=UserWarning)

# 데이터 로드
eval_df = pd.read_csv('/data/my_PRAG/baseline/output_wdoc/evaluation_report.csv')
gen_df = pd.read_csv('/data/my_PRAG/baseline/output_wdoc/generation_report.csv')
idx_df = pd.read_csv('/data/my_PRAG/baseline/output_wdoc/indexing_report.csv')

# 방법 순서 정의
METHOD_ORDER = ['standard', 'random', 'cosine_only', 'pref_cluster_filter', 'naive_p']

# 1. 성능 지표 분석
def analyze_performance():
    performance = eval_df.groupby('method').agg({
        'unhelpful': 'mean',
        'inconsistent': 'mean',
        'hallucination_of_preference_violation': 'mean',
        'preference_unaware_violation': 'mean',
        'preference_following_accuracy(%)': 'mean'
    }).round(2)
    
    # 순서 재정렬
    performance = performance.reindex(METHOD_ORDER)

    print("\n=== 각 방법별 평균 성능 ===")
    print(performance)

    plt.figure(figsize=(12, 6))
    performance['preference_following_accuracy(%)'].plot(kind='bar')
    plt.title('Preference Following Accuracy by Method')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_accuracy.png')
    plt.close()

    error_types = [
        'unhelpful', 'inconsistent',
        'hallucination_of_preference_violation',
        'preference_unaware_violation'
    ]
    plt.figure(figsize=(12, 6))
    performance[error_types].plot(kind='bar', stacked=True)
    plt.title('Error Types Distribution by Method')
    plt.ylabel('Number of Errors')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_types.png')
    plt.close()

# 2. 처리 시간 분석
def analyze_processing_time():
    time_metrics = idx_df.groupby('method').agg({
        'cosine_filter_time(s)': 'mean',
        'random_filter_time(s)': 'mean',
        'cluster_filter_time(s)': 'mean',
        'llm_time(s)': 'mean',
        'summary_time(s)': 'mean',
        'faiss_time(s)': 'mean',
        'total_time(s)': 'mean'
    }).round(2)
    
    # 순서 재정렬
    time_metrics = time_metrics.reindex(METHOD_ORDER)

    print("\n=== 각 방법별 평균 처리 시간 (초) ===")
    print(time_metrics)

    plt.figure(figsize=(12, 6))
    time_metrics[['cosine_filter_time(s)', 'random_filter_time(s)', 'cluster_filter_time(s)', 'llm_time(s)', 'summary_time(s)', 'faiss_time(s)']].plot(
        kind='bar', stacked=True)
    plt.title('Processing Time Distribution by Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('processing_time.png')
    plt.close()

# 3. 검색 성능 분석
def analyze_retrieval():
    retrieval_time = gen_df.groupby('method')['avg_retrieval_time(s)'].mean().round(4)
    
    # 순서 재정렬
    retrieval_time = retrieval_time.reindex(METHOD_ORDER)

    print("\n=== 각 방법별 평균 검색 시간 (초) ===")
    print(retrieval_time)

    # pref_cluster_filter의 시간을 제외한 나머지 방법들의 시간 분포 확인
    other_methods_time = retrieval_time.drop('pref_cluster_filter')
    max_other_time = other_methods_time.max()
    
    plt.figure(figsize=(12, 6))
    # 두 개의 y축 생성
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # pref_cluster_filter를 제외한 방법들
    other_methods_time.plot(kind='bar', ax=ax1, color='blue', alpha=0.7, label='Other Methods')
    ax1.set_ylabel('Time (seconds)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # pref_cluster_filter
    pref_time = retrieval_time['pref_cluster_filter']
    ax2.bar('pref_cluster_filter', pref_time, color='red', alpha=0.7, label='Pref Cluster Filter')
    ax2.set_ylabel('Time (seconds)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # y축 범위 설정
    ax1.set_ylim(0, max_other_time * 1.2)
    ax2.set_ylim(0, pref_time * 1.2)
    
    plt.title('Average Retrieval Time by Method')
    plt.xticks(rotation=45)
    
    # 범례 추가
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('retrieval_time.png')
    plt.close()

# 4. 데이터 처리량 분석
def analyze_data_volume():
    # 각 방법별로 사용할 컬럼 매핑
    method_columns = {
        'cosine_only': ['cosine_kept'],
        'random': ['random_kept'],
        'naive_p': ['kept', 'summarized'],
        'pref_cluster_filter': ['cluster_kept'],
        'standard': ['standard_kept']
    }
    
    # 결과를 저장할 데이터프레임 생성
    result_data = {}
    
    # 각 방법별로 데이터 처리
    for method, columns in method_columns.items():
        if method == 'standard':
            result_data[method] = {'standard_kept': 1581937}
        else:
            method_data = idx_df[idx_df['method'] == method][columns].mean()
            result_data[method] = method_data.to_dict()
    
    # 결과 데이터프레임 생성
    data_volume = pd.DataFrame(result_data).T.round(2)
    
    # 순서 재정렬
    data_volume = data_volume.reindex(METHOD_ORDER)
    
    print("\n=== 각 방법별 평균 데이터 처리량 ===")
    print(data_volume)

    # 시각화를 위한 데이터 준비
    plot_data = pd.DataFrame(index=METHOD_ORDER)
    
    for method in METHOD_ORDER:
        if method == 'standard':
            plot_data.loc[method, 'standard_kept'] = 1581937
        elif method == 'naive_p':
            plot_data.loc[method, 'kept'] = data_volume.loc[method, 'kept']
            plot_data.loc[method, 'summarized'] = data_volume.loc[method, 'summarized']
        else:
            col = list(method_columns[method])[0]
            plot_data.loc[method, col] = data_volume.loc[method, col]

    # 전체 단계별 chunk 수 시각화
    plt.figure(figsize=(12, 6))
    plot_data.plot(kind='bar', stacked=True)
    plt.title('Data Processing Volume by Method')
    plt.ylabel('Number of Chunks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data_volume_stacked.png')
    plt.close()

# 5. 종합 분석 보고서
def generate_summary_report():
    print("\n=== 종합 분석 보고서 ===")

    performance_rank = eval_df.groupby('method')['preference_following_accuracy(%)'] \
                              .mean().reindex(METHOD_ORDER)
    print("\n1. 선호도 준수 정확도:")
    print(performance_rank)

    time_rank = idx_df.groupby('method')['total_time(s)'].mean().reindex(METHOD_ORDER)
    print("\n2. 총 처리 시간 (초):")
    print(time_rank)

    retrieval_rank = gen_df.groupby('method')['avg_retrieval_time(s)'].mean().reindex(METHOD_ORDER)
    print("\n3. 평균 검색 시간 (초):")
    print(retrieval_rank)

    efficiency = idx_df.groupby('method').agg({
        'kept': 'mean',
        'total_time(s)': 'mean'
    }).reindex(METHOD_ORDER)
    efficiency['chunks_per_second'] = (efficiency['kept'] / efficiency['total_time(s)']).round(2)
    print("\n4. 초당 처리 청크 수:")
    print(efficiency['chunks_per_second'])

# 메인 실행
if __name__ == "__main__":
    analyze_performance()
    analyze_processing_time()
    analyze_retrieval()
    analyze_data_volume()
    generate_summary_report()
