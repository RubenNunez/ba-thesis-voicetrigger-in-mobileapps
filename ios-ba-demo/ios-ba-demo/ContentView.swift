//
//  ContentView.swift
//  ios-ba-demo
//
//  Created by Ruben Nunez on 06.11.23.
//

import SwiftUI

struct ContentView: View {
    //@ObservedObject var audioListener = AudioListener()
    @ObservedObject var audioDataManager = AudioDataManager() // V2
    
    var body: some View {
        VStack {
            WaveformView(samples: self.audioDataManager.lastCapturedData)
                
            if audioDataManager.isStreaming {
                Button(action: {
                    self.audioDataManager.stopAudioCapture()
                }) {
                    Image(systemName: "stop.circle.fill")
                        .resizable()
                        .frame(width: 64, height: 64)
                        .foregroundColor(.red)
                        .padding()
                }
            } else {
                Button(action: {
                    self.audioDataManager.startAudioCapture()
                }) {
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .frame(width: 64, height: 64)
                        .foregroundColor(.green)
                        .padding()
                }
                
            }
        }
        .padding()
        .sheet(item: $audioDataManager.fileURL) { identifiableURL in
            DocumentPickerView(fileURL: identifiableURL.url)
               }
    }
}



struct DocumentPickerView: UIViewControllerRepresentable {
    var fileURL: URL

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forExporting: [fileURL])
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {
        // You can update the picker if required here
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
