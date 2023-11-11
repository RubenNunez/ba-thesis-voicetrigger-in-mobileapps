import torch
from model import WakeupTriggerConvLSTM2s


base_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-best/"
model_path = base_path + "checkpoint_epoch_92_loss_0.044720325733640424.pt"

# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 256]
example_input = torch.rand(1, 1, 128, 256)  

model = WakeupTriggerConvLSTM2s(device="cpu")
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


traced_model = torch.jit.trace(model, example_input)

traced_model.save("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-best/model_trace.ptl")
