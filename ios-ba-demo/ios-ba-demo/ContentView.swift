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
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
