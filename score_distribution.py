import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mydata_utils import MyDataUtils

class ScoreDistributionAnalyzer(MyDataUtils):
    def __init__(self, utils):
        super().__init__(
            mode=utils.mode,
            method=utils.method,
            device=utils.device,
            use_multi_gpu=utils.use_multi_gpu,
            chunk_mode=utils.chunk_mode,
            output_dir=utils.output_dir,
            persona_task_file=utils.persona_task_file,
            emb_model_name=utils.emb_model_name
        )
        self.utils = utils

    def analyze_score_distribution(self, persona_index):
        print(f"\n=== Starting score distribution analysis for persona {persona_index} ===")
        
        # 데이터 로드
        persona = self.load_persona_data(persona_index)
        self.load_models()

        # chunks와 embeddings 로드
        print("Loading chunks and embeddings...")
        with open(self.chunk_file, "r", encoding="utf-8") as f:
            chunks = [json.loads(line)["text"] for line in f]
        chunk_embeddings = np.load(self.embedding_file)
        chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # 선호/불호 분리
        processed_preferences = []
        for preference in [block["preference"] for block in persona["preference_blocks"]]:
            likes_dislikes = self.extract_likes_dislikes(preference)
            processed_preferences.append({
                "likes": likes_dislikes["likes"],
                "dislikes": likes_dislikes["dislikes"]
            })

        # 선호와 불호 임베딩 생성
        like_texts = [p["likes"] for p in processed_preferences]
        dislike_texts = [p["dislikes"] for p in processed_preferences]
        
        like_embeddings = self.embed_texts(like_texts)
        dislike_embeddings = self.embed_texts(dislike_texts)
        
        like_embeddings = like_embeddings / np.linalg.norm(like_embeddings, axis=1, keepdims=True)
        dislike_embeddings = dislike_embeddings / np.linalg.norm(dislike_embeddings, axis=1, keepdims=True)

        # beta 값 범위 설정
        beta_values = np.arange(0, 1.6, 0.1)  # 0부터 1.5까지 0.1 단위로 15개 값
        alpha = 1.0  # 고정된 alpha 값
        LIKE_THRESHOLD = 0.7  # 고정된 LIKE_THRESHOLD 값

        # 결과 저장을 위한 디렉토리 생성
        output_dir = os.path.join(self.output_dir, f"score_analysis/{persona_index}")
        os.makedirs(output_dir, exist_ok=True)

        # 배치 단위로 처리
        batch_size = self.batch_size
        all_scores = []

        for beta in tqdm(beta_values, desc="Analyzing different beta values"):
            batch_scores = []
            
            for i in range(0, len(chunk_embeddings_norm), batch_size):
                batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
                
                # 선호와 불호에 대한 유사도 계산
                like_sims = np.dot(like_embeddings, batch_embeddings.T)
                dislike_sims = np.dot(dislike_embeddings, batch_embeddings.T)
                
                # 각 선호/불호 페어별로 스코어 계산
                pair_scores = []
                for like_sim, dislike_sim in zip(like_sims, dislike_sims):
                    if np.any(like_sim > LIKE_THRESHOLD):
                        pair_score = alpha * np.max(like_sim) - beta * np.max(dislike_sim)
                    else:
                        pair_score = 0
                    pair_scores.append(pair_score)
                
                pair_scores = np.array(pair_scores)
                if len(pair_scores) > 0:
                    scores = np.max(pair_scores, axis=0)
                    if np.isscalar(scores):
                        scores = np.array([scores])
                else:
                    scores = np.array([0])
                # 0이 아닌 스코어만 저장
                batch_scores.extend(scores[scores != 0])
            
            all_scores.append(batch_scores)

        # 시각화
        plt.figure(figsize=(15, 10))
        
        # 1. 박스플롯
        plt.subplot(2, 1, 1)
        plt.boxplot(all_scores, tick_labels=[f'β={beta:.1f}' for beta in beta_values])
        plt.title('Score Distribution by Beta Value (Non-zero scores only)')
        plt.xlabel('Beta Value')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # 2. 히스토그램
        plt.subplot(2, 1, 2)
        max_density = 200  # 최대 density 값 제한
        for i, scores in enumerate(all_scores):
            if len(scores) > 0 and np.std(scores) > 0:  # 분산이 있는 경우에만 KDE 플롯
                sns.kdeplot(scores, label=f'β={beta_values[i]:.1f}', alpha=0.5)
            else:  # 분산이 없는 경우 히스토그램으로 표시
                plt.axvline(x=np.mean(scores), label=f'β={beta_values[i]:.1f} (mean={np.mean(scores):.2f})', alpha=0.5)
        plt.title('Score Density by Beta Value (Non-zero scores only)')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.ylim(0, max_density)  # y축 범위 제한
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
        plt.close()

        # 통계 정보 저장
        stats = {
            'beta_values': beta_values.tolist(),
            'statistics': []
        }
        
        for i, scores in enumerate(all_scores):
            if len(scores) > 0:  # 빈 배열이 아닌 경우에만 통계 계산
                stats['statistics'].append({
                    'beta': float(beta_values[i]),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores)),
                    'q1': float(np.percentile(scores, 25)),
                    'q3': float(np.percentile(scores, 75)),
                    'non_zero_count': len(scores)
                })
            else:  # 빈 배열인 경우
                stats['statistics'].append({
                    'beta': float(beta_values[i]),
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                    'q1': 0.0,
                    'q3': 0.0,
                    'non_zero_count': 0
                })
        
        with open(os.path.join(output_dir, 'score_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Analysis completed. Results saved in {output_dir}")

    def analyze_like_threshold_effect(self, persona_index):
        print(f"\n=== Starting LIKE_THRESHOLD analysis for persona {persona_index} ===")
        
        # 데이터 로드
        persona = self.load_persona_data(persona_index)
        self.load_models()

        # chunks와 embeddings 로드
        print("Loading chunks and embeddings...")
        with open(self.chunk_file, "r", encoding="utf-8") as f:
            chunks = [json.loads(line)["text"] for line in f]
        chunk_embeddings = np.load(self.embedding_file)
        chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # 선호/불호 분리
        processed_preferences = []
        for preference in [block["preference"] for block in persona["preference_blocks"]]:
            likes_dislikes = self.extract_likes_dislikes(preference)
            processed_preferences.append({
                "likes": likes_dislikes["likes"],
                "dislikes": likes_dislikes["dislikes"]
            })

        # 선호와 불호 임베딩 생성
        like_texts = [p["likes"] for p in processed_preferences]
        dislike_texts = [p["dislikes"] for p in processed_preferences]
        
        like_embeddings = self.embed_texts(like_texts)
        dislike_embeddings = self.embed_texts(dislike_texts)
        
        like_embeddings = like_embeddings / np.linalg.norm(like_embeddings, axis=1, keepdims=True)
        dislike_embeddings = dislike_embeddings / np.linalg.norm(dislike_embeddings, axis=1, keepdims=True)

        # LIKE_THRESHOLD 값 범위 설정
        like_threshold_values = np.arange(0.3, 0.95, 0.05)  # 0.3부터 0.9까지 0.05 단위
        alpha = 1.0  # 고정된 alpha 값
        beta = 1.0   # 고정된 beta 값
        THRESHOLD = 0.3  # 고정된 THRESHOLD 값

        # 결과 저장을 위한 디렉토리 생성
        output_dir = os.path.join(self.output_dir, f"like_threshold_analysis/{persona_index}")
        os.makedirs(output_dir, exist_ok=True)

        # 배치 단위로 처리
        batch_size = self.batch_size
        threshold_results = []

        for like_threshold in tqdm(like_threshold_values, desc="Analyzing different LIKE_THRESHOLD values"):
            kept_count = 0
            total_processed = 0
            
            for i in range(0, len(chunk_embeddings_norm), batch_size):
                batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]
                
                # 선호와 불호에 대한 유사도 계산
                like_sims = np.dot(like_embeddings, batch_embeddings.T)
                dislike_sims = np.dot(dislike_embeddings, batch_embeddings.T)
                
                # 각 선호/불호 페어별로 스코어 계산
                pair_scores = []
                for like_sim, dislike_sim in zip(like_sims, dislike_sims):
                    if np.any(like_sim > like_threshold):
                        pair_score = alpha * np.max(like_sim) - beta * np.max(dislike_sim)
                    else:
                        pair_score = 0
                    pair_scores.append(pair_score)
                
                pair_scores = np.array(pair_scores)
                if len(pair_scores) > 0:
                    scores = np.max(pair_scores, axis=0)
                    if np.isscalar(scores):
                        scores = np.array([scores])
                else:
                    scores = np.array([0])
                
                # THRESHOLD를 넘는 문서 개수 계산
                kept_count += np.sum(scores > THRESHOLD)
                total_processed += len(scores)
            
            threshold_results.append({
                'like_threshold': float(like_threshold),
                'kept_count': int(kept_count),
                'total_count': int(total_processed),
                'kept_ratio': float(kept_count / total_processed) if total_processed > 0 else 0.0
            })

        # 시각화
        plt.figure(figsize=(12, 8))
        
        # 1. 문서 개수 변화
        plt.subplot(2, 1, 1)
        like_thresholds = [r['like_threshold'] for r in threshold_results]
        kept_counts = [r['kept_count'] for r in threshold_results]
        plt.plot(like_thresholds, kept_counts, 'b-o', linewidth=2, markersize=6)
        plt.title('Number of Kept Documents by LIKE_THRESHOLD')
        plt.xlabel('LIKE_THRESHOLD')
        plt.ylabel('Number of Kept Documents')
        plt.grid(True, alpha=0.3)
        
        # 2. 비율 변화
        plt.subplot(2, 1, 2)
        kept_ratios = [r['kept_ratio'] * 100 for r in threshold_results]
        plt.plot(like_thresholds, kept_ratios, 'r-o', linewidth=2, markersize=6)
        plt.title('Percentage of Kept Documents by LIKE_THRESHOLD')
        plt.xlabel('LIKE_THRESHOLD')
        plt.ylabel('Percentage of Kept Documents (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'like_threshold_analysis.png'))
        plt.close()

        # 결과 저장
        with open(os.path.join(output_dir, 'like_threshold_results.json'), 'w') as f:
            json.dump(threshold_results, f, indent=2)

        print(f"LIKE_THRESHOLD analysis completed. Results saved in {output_dir}")
        print(f"Total documents: {threshold_results[0]['total_count']}")
        print("LIKE_THRESHOLD vs Kept documents:")
        for result in threshold_results:
            print(f"  {result['like_threshold']:.2f}: {result['kept_count']} documents ({result['kept_ratio']*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_index", type=str, required=True, help="Persona index (0-10) or 'all'")
    args = parser.parse_args()

    indices = list(range(10)) if args.persona_index == "all" else [int(args.persona_index)]
    from mydata_utils import MyDataUtils
    
    # MyDataUtils 초기화에 필요한 파라미터들
    utils = MyDataUtils(
        mode="indexing",
        method="score_p",
        device="cuda",
        use_multi_gpu=False,
        chunk_mode="wdoc",
        output_dir="./output",
        persona_task_file="final_persona_tasks.json",
        emb_model_name="facebook/contriever"
    )
    
    analyzer = ScoreDistributionAnalyzer(utils)
    for persona_index in indices:
        analyzer.analyze_score_distribution(persona_index=persona_index)
        analyzer.analyze_like_threshold_effect(persona_index=persona_index) 