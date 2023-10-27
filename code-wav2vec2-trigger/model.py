from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
import torch.nn as nn

config = Wav2Vec2Config.from_pretrained("../wav2vec2-base-960h")

# Create a new custom model based on the Wav2Vec2Model (not Wav2Vec2ForCTC)
class TriggerWordWav2Vec2Model(nn.Module):
    def __init__(self, config):
        super(TriggerWordWav2Vec2Model, self).__init__()
        self.wav2vec2 = Wav2Vec2ForCTC(config).wav2vec2
        # The new output layer: binary classification
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_values):
        # Get the transformer output
        transformer_outputs = self.wav2vec2(input_values).last_hidden_state
        # Average the transformer outputs along the time dimension (dim=1)
        avg_pool = transformer_outputs.mean(dim=1)
        # Pass through the new classifier layer
        return self.classifier(avg_pool)
