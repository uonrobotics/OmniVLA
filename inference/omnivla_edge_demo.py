import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
import clip

# OmniVLA-edge 필수 유틸리티
sys.path.insert(0, '..')
from utils_policy import load_model, transform_images_PIL_mask, transform_images_map

# ===============================================================
# 전역 설정 변수 (여기서 모드를 변경하세요)
# ===============================================================
# 7: Language Only (언어 명령 기반 주행)
# 6: Image Goal Only (목적지 사진 기반 주행)
# 4: Pose Only (좌표 기반 주행)
MODALITY_ID = 7

LAN_PROMPT = "Go to the white paper with the number 1 and stop."
GOAL_IMAGE_PATH = "inference/goal_img.jpg"
# ===============================================================

class OmniVLAEdgeConfigNode(Node):
    def __init__(self):
        super().__init__("omnivla_config_node")
        
        # 1. 모델 파라미터 설정
        self.model_params = {
            "model_type": "omnivla-edge", "len_traj_pred": 8, "learn_angle": True,
            "context_size": 5, "obs_encoder": "efficientnet-b0", "encoding_size": 256,
            "obs_encoding_size": 1024, "goal_encoding_size": 1024, "late_fusion": False,
            "mha_num_attention_heads": 4, "mha_num_attention_layers": 4,
            "mha_ff_dim_factor": 4, "clip_type": "ViT-B/32"
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsize = (96, 96)
        self.imgsize_clip = (224, 224)
        
        # 2. 모델 초기화
        self.setup_model()
        
        # 3. 상태 변수 및 마스크
        self.current_image_pil = None
        self.context_queue = []
        self.mask_360_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_360_224 = np.ones((224, 224, 3), dtype=np.float32)

        # 목적지 이미지 미리 로드 (ID 6 등을 위해)
        self.goal_image_tensor = self.load_goal_image(GOAL_IMAGE_PATH)

        # 4. ROS 통신 설정
        self.sub_img = self.create_subscription(
            CompressedImage, "/act/input/front_rgb/compressed", self.cb_image, 10)
        self.pub_cmd = self.create_publisher(
            TwistStamped, "/act/output/cmd_vel", 10)
        
        self.create_timer(0.33, self.inference_loop)
        self.get_logger().info(f"🚀 OmniVLA-edge Node Started with Modality ID: {MODALITY_ID}")

    def setup_model(self):
        ckpth_path = "./omnivla-edge/omnivla-edge.pth"
        self.model, self.text_encoder, _ = load_model(ckpth_path, self.model_params, self.device)
        self.model.to(self.device).eval()
        self.text_encoder.to(self.device).eval()

    def load_goal_image(self, path):
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize(self.imgsize)
            return transform_images_PIL_mask(img, self.mask_360_96).to(self.device)
        return None

    def cb_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.current_image_pil = Image.fromarray(cv_img_rgb)

    def inference_loop(self):
        if self.current_image_pil is None:
            return

        # 1. 이미지 전처리
        img_96 = self.current_image_pil.resize(self.imgsize)
        img_224 = self.current_image_pil.resize(self.imgsize_clip)
        
        if len(self.context_queue) < self.model_params["context_size"] + 1:
            self.context_queue = [img_96] * (self.model_params["context_size"] + 1)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(img_96)

        # 2. 텐서 준비
        obs_images = transform_images_PIL_mask(self.context_queue, self.mask_360_96).to(self.device)
        obs_images_split = torch.split(obs_images, 3, dim=1)
        obs_image_cur = obs_images_split[-1]
        obs_images_cat = torch.cat(obs_images_split, dim=1)
        cur_large_img = transform_images_PIL_mask(img_224, self.mask_360_224).to(self.device)
        
        dummy_map = transform_images_map(Image.new("RGB", (352, 352), color=(0, 0, 0))).to(self.device)
        map_images = torch.cat((dummy_map, dummy_map, obs_image_cur), axis=1)

        # 3. 입력 데이터 구성 (Modality에 따라 동적 할당)
        modality_tensor = torch.tensor([MODALITY_ID]).to(self.device)
        
        # 목적지 이미지 (ID 6/9가 아니면 현재 이미지로 무시 처리)
        g_img = self.goal_image_tensor if (MODALITY_ID in [6, 9] and self.goal_image_tensor is not None) else transform_images_PIL_mask(img_96, self.mask_360_96).to(self.device)

        # 언어 인코딩 (ID 7/8/9가 아니면 더미 텍스트 사용)
        prompt = LAN_PROMPT if MODALITY_ID in [7, 8, 9] else "xxxx"
        feat_text_lan = self.text_encoder.encode_text(clip.tokenize(prompt, truncate=True).to(self.device)).to(self.device)

        # 4. 모델 추론
        with torch.no_grad():
            actions, _, _ = self.model(
                obs_images_cat, 
                torch.tensor([[0.0, 0.0, 1.0, 0.0]]).to(self.device).float(), # Dummy Pose
                map_images, 
                g_img, 
                modality_tensor, 
                feat_text_lan, 
                cur_large_img
            )

        # 5. 제어 명령 생성
        waypoints = actions.float().cpu().numpy()[0]
        v, w = self.calculate_pd_control(waypoints[2])
        self.publish_command(v, w)

    def calculate_pd_control(self, waypoint):
        # 1. 좌표 변환 (0.1은 spacing)
        dx = waypoint[0] * 0.1  # 전방 거리
        dy = waypoint[1] * 0.1  # 측면 거리
        
        # 2. 거리와 각도 계산
        dist = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)
        
        # ---------------------------------------------------------
        # 3. 진동 방지 튜닝 (이 부분값을 수정하세요!)
        # ---------------------------------------------------------
        # v_gain: 거리당 속도 (0.5 ~ 1.0 사이 추천)
        # w_gain: 각도 오차당 회전 속도 (진동이 심하면 이 값을 낮추세요! 0.5 ~ 1.0 추천)
        v_gain = 0.6  
        w_gain = 0.7  # <--- 진동이 심하면 0.5 이하로 더 낮추세요
        
        v = dist * v_gain
        w = target_angle * w_gain
        
        # 4. 근접 시 강제 감속 (Dead-zone)
        # 목표물과 15cm 이내로 가까워지면 진동 방지를 위해 회전을 억제합니다.
        if dist < 0.15:
            w *= 0.5 
            if dist < 0.05: # 5cm 이내면 정지
                return 0.0, 0.0

        # 5. 최종 하드웨어 제한 (매우 보수적으로 설정)
        max_v = 0.10  # 초당 10cm (천천히 접근)
        max_w = 0.4   # 초당 약 23도 회전 제한
        
        v = np.clip(v, 0.0, max_v)
        w = np.clip(w, -max_w, max_w)
        
        self.get_logger().info(f"Target({dx:.2f}, {dy:.2f}) | Dist: {dist:.2f} | v: {v:.2f}, w: {w:.2f}")
        return v, w

    def publish_command(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(-w)
        self.pub_cmd.publish(msg)

def main():
    rclpy.init()
    node = OmniVLAEdgeConfigNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()