from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
import torch.nn as nn

config = Wav2Vec2Config.from_pretrained("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/wav2vec2-base-960h")

class TriggerWordWav2Vec2Model(nn.Module):
    def __init__(self, config):
        super(TriggerWordWav2Vec2Model, self).__init__()
        self.wav2vec2 = Wav2Vec2ForCTC(config).wav2vec2
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_values):
        transformer_outputs = self.wav2vec2(input_values).last_hidden_state
        avg_pool = transformer_outputs.mean(dim=1)
        return self.classifier(avg_pool)
