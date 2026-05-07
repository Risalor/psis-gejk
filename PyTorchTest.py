# diagnose_gpu.py
import os
import sys
sys.path.append('/home/risalor/Desktop/psisShit/psis-gejk')
sys.path.append('/home/risalor/Desktop/psisShit/psis-gejk/venv/lib/python3.14/site-packages')

import torch
import gymnasium as gym
from rl_agents.agents.common.utils import choose_device

print("=== System Info ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
print("\n=== Testing choose_device ===")
result = choose_device("cuda:0", default_device="cpu")
print(f"choose_device('cuda:0') returned: {result}")
print(f"Result type: {type(result)}")

# Check if it's actually a GPU device
if hasattr(result, 'type'):
    print(f"Is cuda device: {result.type == 'cuda'}")
else:
    print(f"Is cuda string: {'cuda' in str(result)}")

# Test model creation
print("\n=== Testing Model on GPU ===")
from rl_agents.agents.common.models import model_factory

# Create a minimal env config
class DummyEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)

env = DummyEnv()
config = {
    "model": {
        "type": "DuelingNetwork",
        "in": 4
    },
    "device": "cuda:0"
}

model = model_factory(config["model"])
print(f"Model device: {next(model.parameters()).device}")

try:
    model = model.to(result)
    print(f"After .to(device), model device: {next(model.parameters()).device}")
    
    # Test forward pass
    test_input = torch.randn(1, 4).to(result)
    output = model(test_input)
    print(f"Forward pass successful! Output shape: {output.shape}")
    print(f"Output device: {output.device}")
except Exception as e:
    print(f"GPU forward pass failed: {e}")