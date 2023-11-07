//
//  ContentView.swift
//  ios-ba-demo
//
//  Created by Ruben Nunez on 06.11.23.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var audioListener = AudioListener()
    var body: some View {
        VStack {
            WaveformView(samples: audioListener.latestSamples)
                
            if audioListener.isStreaming {
                Button(action: {
                    self.audioListener.stopStreaming()
                }) {
                    Image(systemName: "stop.circle.fill")
                        .resizable()
                        .frame(width: 64, height: 64)
                        .foregroundColor(.red)
                        .padding()
                }
            } else {
                Button(action: {
                    self.audioListener.startStreaming()
                }) {
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .frame(width: 64, height: 64)
                        .foregroundColor(.green)
                        .padding()
                }
            }
            
            /*Button(action: {
                self.audioListener.recordTwoSecondClip { clip in
                    self.audioListener.saveAudioToFile(samples: clip)
                }
            }){
                Text("Record")
            }*/

        }
        .padding()
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
