import torch
print(torch.__version__)

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"현재 CUDA 버전: {torch.version.cuda}")
print(f"GPU 개수: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"현재 GPU: {torch.cuda.get_device_name(0)}")