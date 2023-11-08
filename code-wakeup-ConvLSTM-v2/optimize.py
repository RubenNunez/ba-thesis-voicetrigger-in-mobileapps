import torch
import torch.nn as nn
import torch.utils.mobile_optimizer as mobile_optimizer
from model import WakeupTriggerConvLSTM2s
from transform import AudioToSpectrogramTransformJit

base_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/"
model_path = base_path + "checkpoint_epoch_42_loss_0.010789588726497367.pt"

torch.backends.quantized.engine = 'qnnpack'  # set quantization engine

# Load transform jit
scripted_transform = torch.jit.script(AudioToSpectrogramTransformJit())

# Load model
model = WakeupTriggerConvLSTM2s(device="cpu")
checkpoint = torch.load(model_path) #, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Quantize the model
# quantized_model = torch.quantization.quantize_dynamic(
#     model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
# )

# Script the model
scripted_model = torch.jit.script(model) # quantized_model

# Optimize the model for mobile
# optimized_model = mobile_optimizer.optimize_for_mobile(scripted_model)

# Save model for Lite Interpreter
scripted_model._save_for_lite_interpreter("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/optimized_model.ptl")

# Save transform for Lite Interpreter
scripted_transform._save_for_lite_interpreter("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/transform_jit.ptl")
