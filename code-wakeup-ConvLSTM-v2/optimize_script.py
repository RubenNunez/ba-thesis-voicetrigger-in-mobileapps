import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import WakeupTriggerConvLSTM2s
from transform import AudioToSpectrogramTransformJit

base_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/"
model_path = base_path + "checkpoint_epoch_3_loss_0.26515502861744705.pt"

# Load model
model = WakeupTriggerConvLSTM2s(device="cpu")
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Script the model
scripted_model = torch.jit.script(model)

# Optimize for mobile
# optimized_scripted_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)

# Save optimized scripted model for Lite Interpreter
scripted_model = optimize_for_mobile(scripted_model)
scripted_model._save_for_lite_interpreter("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/model.ptl")

# Load transform jit
transform = AudioToSpectrogramTransformJit()
transform.eval()
scripted_transform = torch.jit.script(transform)

# Save scripted transform for Lite Interpreter
scripted_transform = optimize_for_mobile(scripted_transform)
scripted_transform._save_for_lite_interpreter("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/transform.ptl")
