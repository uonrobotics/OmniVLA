import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm

# 기존 프로젝트 경로 추가 (상황에 맞게 수정)
import sys
sys.path.append("../lerobot/src/")

from prismatic.vla.datasets.uonamr_dataset import UONAMR_Dataset
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from transformers import AutoProcessor

# ==========================================
# 1. 환경 설정 (Inference/Train 설정과 동일하게)
# ==========================================
VLA_PATH = "openvla/openvla-7b" # 혹은 로컬 경로
DATA_ROOT = Path("/home/sujin/workspace/physical-ai/OmniVLA/dataset/data_goto")

def get_dataset():
    processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # 데이터셋 인스턴스 생성
    dataset = UONAMR_Dataset(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        root=DATA_ROOT,
    )
    return dataset

# ==========================================
# 2. 시각화 핵심 함수
# ==========================================
def visualize_batch_debug(dataset, num_samples=5):
    # 테스트할 랜덤 인덱스 추출
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        print(f"\n--- Visualizing Index: {idx} ---")
        sample = dataset[idx]
        
        # 데이터 추출
        curr_pil = sample["img_PIL"]
        goal_pil = sample["gimg_PIL"]
        actions = sample["actions"].numpy()  # (8, 4) -> [dx, dy, cos, sin]
        lan_prompt = sample["lan_prompt"]
        mod_id = sample["modality_id"]
        
        # 시각화 레이아웃 생성
        fig = plt.figure(figsize=(22, 10), facecolor='white')
        gs = fig.add_gridspec(2, 3)
        
        # (1) 현재 관측 (Current Image)
        ax_curr = fig.add_subplot(gs[0, 0])
        ax_curr.imshow(curr_pil)
        ax_curr.set_title(f"CURRENT (Idx: {idx})\nModality: {mod_id}", fontsize=12)
        ax_curr.axis('off')

        # (2) 목표 관측 (Goal Image & Masking Check)
        ax_goal = fig.add_subplot(gs[1, 0])
        # Modality 7(Language Only)일 때 이미지가 0인지 확인
        pixel_goal_sum = sample["pixel_values_goal"].sum()
        if pixel_goal_sum == 0:
            ax_goal.imshow(np.zeros((224, 224, 3)))
            ax_goal.set_title("GOAL (MASKED for Language Mode)", color='red')
        else:
            ax_goal.imshow(goal_pil)
            ax_goal.set_title(f"GOAL (Target Position)\n{lan_prompt}", fontsize=10)
        ax_goal.axis('off')

        # (3) 로봇 경로 (Trajectory - Top-down View)
        ax_traj = fig.add_subplot(gs[:, 1:])
        
        # 1. 로봇 시작 위치 (검은색 사각형 크기 축소)
        ax_traj.scatter(0, 0, color='black', marker='s', s=50, label='Robot Start', zorder=5)
        
        # 2. 로봇 현재 방향 지시 화살표 (크기를 훨씬 작게 조정)
        # 로봇이 0.25m/s이므로, 초기 방향 화살표 길이를 0.05m(5cm) 정도로 줄입니다.
        ax_traj.arrow(0, 0, 0, 0.05, head_width=0.015, head_length=0.02, 
                      fc='black', ec='black', zorder=6)
        
        # Waypoints: X(전진), Y(좌측+)
        m_spacing = dataset.metric_spacing # 0.05 기준
        x_fwd = actions[:, 0] * m_spacing
        y_lat = -actions[:, 1] * m_spacing 
        
        # 3. 경로 선 (두께와 마커 크기 최적화)
        ax_traj.plot(y_lat, x_fwd, 'b-o', alpha=0.4, linewidth=2, markersize=6, 
                     label='Action Waypoints', zorder=3)
        
        # 4. 각 웨이포인트의 Heading(녹색 화살표) 최적화
        # 화살표가 점을 가리지 않도록 길이를 아주 짧게(3cm) 조정합니다.
        for i in range(len(actions)):
            dx_h = -actions[i, 3] * 0.03 # sin (방향 성분)
            dy_h = actions[i, 2] * 0.03  # cos (방향 성분)
            
            # 화살표 머리 크기를 아주 작게 설정하여 궤적 선이 보이게 함
            ax_traj.arrow(y_lat[i], x_fwd[i], dx_h, dy_h, 
                          head_width=0.008, head_length=0.01, 
                          fc='green', ec='green', alpha=0.6, zorder=4)

        # 5. 그래프 범위 설정 (0.25m/s 로봇의 2.6초 이동 거리인 0.6~0.7m에 최적화)
        ax_traj.set_xlim(-0.4, 0.4) # 좌우 40cm
        ax_traj.set_ylim(-0.05, 0.8) # 전방 80cm (총 궤적이 약 0.66m이므로 적당함)
        ax_traj.set_aspect('equal')
        ax_traj.grid(True, linestyle='--', alpha=0.5)

        # 텍스트 정보 표시
        # ActionTokenizer가 아닌 base_tokenizer(혹은 processor.tokenizer)를 사용하여 디코딩합니다.
        valid_label_ids = sample['labels'][sample['labels'] != -100]
        decoded_text = dataset.base_tokenizer.decode(valid_label_ids)

        info_text = (
            f"Action Tokens (First): {decoded_text[:50]}...\n"
            f"Final Goal Distance: {sample['temp_dist']} steps"
        )
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.1))

        plt.tight_layout()
        plt.show()

# ==========================================
# 3. 실행부
# ==========================================
import traceback

if __name__ == "__main__":
    try:
        uon_dataset = get_dataset()
        print(f"Dataset Loaded. Total frames: {len(uon_dataset)}")
        
        # 시각화 실행
        visualize_batch_debug(uon_dataset, num_samples=30)
        
    except Exception as e:
        traceback.print_exc()