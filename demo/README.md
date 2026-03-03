
conda create -n omnivla python=3.10 -y
conda activate omnivla

pip install packaging ninja numpy==1.26.4
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes accelerate
pip install -e ..
pip install "flash-attn==2.5.5" --no-build-isolation
pip install opencv-python pillow
pip install rclpy --extra-index-url https://rospypi.github.io/simple/