import torchaudio
import torch
import os
import random

from model import WakeupModel_LSTM
from dataset import get_featurizer

class SingleFileTester:

    def __init__(self, model_file):
        #self.model = torch.jit.load(model_file) # jit 

        model_params = {
            "num_classes": 1, "feature_size": 40, "hidden_size": 16,
            "num_layers": 1, "dropout" :0.1, "bidirectional": False
        }
        self.model = WakeupModel_LSTM(**model_params, device='cpu')   # Initialize the model first
        
        checkpoint = torch.load(model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().to('cpu')
        
        self.featurizer = get_featurizer(8000)

    def predict(self, audio_file):
        waveform, _ = torchaudio.load(audio_file, normalize=True)
        mfcc_tensor = self.featurizer(waveform)
        mfcc_transposed = mfcc_tensor.transpose(1, 2).transpose(0, 1)
        out = self.model(mfcc_transposed)
        pred = torch.round(torch.sigmoid(out))
        return pred.item()

def test_multiple_files(directory, n_files=2):
    # Get all wav files from directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    
    # Randomly select n_files
    test_files = random.sample(all_files, n_files)
    
    tester = SingleFileTester("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM/checkpoints/wakeup_2023-10-25_18.pt") #optimized/optimized_model.pt")   
    
    for file in test_files:
        prediction = tester.predict(file)
        # print prediction as percentage with 2 decimals
        print(f"Prediction for {os.path.basename(file)}: {(prediction * 100):.2f}%")

if __name__ == "__main__":
    test_directory = '/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM/other/'
    test_directory = '/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM/Hey_FOOBY/'
    
    test_multiple_files(test_directory, 5)
