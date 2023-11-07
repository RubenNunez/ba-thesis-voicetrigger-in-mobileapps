//
//  AudioRecorder.swift
//  ios-ba-demo
//
//  Created by Ruben Nunez on 06.11.23.
//

import Foundation
import AVFoundation

class AudioListener: ObservableObject {
    
    @Published var latestSamples: [Float] = []
    @Published var isStreaming: Bool = false
    
    
    private var audioEngine = AVAudioEngine()
    private var sampleBuffer: [Float] = []
    private let sampleRate: Double = 16000
    private let bufferDuration: Double = 2.0
    private let bufferTimeInterval: Double = 0.1
    private var requiredSamplesCount: Int {
        return Int(sampleRate * bufferDuration)
    }
    
    private lazy var module: TorchModule = {
        if let modelFilePath = Bundle.main.path(forResource: "model", ofType: "ptl"),
           let transformFilePath = Bundle.main.path(forResource: "transform", ofType: "ptl"),
           let module = TorchModule(modelPath: modelFilePath, andTransformPath: transformFilePath) {
            return module
        } else {
            fatalError("Can't find the model or transform file!")
        }
    }()
    
    
    func startStreaming() {
        sampleBuffer = Array(repeating: 0.0, count: requiredSamplesCount)
        // Initialize with zeros (2s width)
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        // Setup callback for processing audio samples
        inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(sampleRate * bufferTimeInterval), format: recordingFormat) { [weak self] (buffer, time) in
            guard let self = self else { return }
            
            let dataPointer = buffer.floatChannelData![0]
            let arr = Array(UnsafeBufferPointer(start: dataPointer, count: Int(buffer.frameLength)))
            
            let minVal = arr.min()
            let maxVal = arr.max()
            print("Min Value: \(minVal ?? 0) Max Value: \(maxVal ?? 0)")

            
            // Roll the sample buffer to remove old audio and append new audio
            self.sampleBuffer = Array(self.sampleBuffer.dropFirst(arr.count))
            self.sampleBuffer.append(contentsOf: arr)
            
            // Pass the buffer for inference
            self.sampleBuffer.withUnsafeMutableBufferPointer { pointer in
                if let baseAddress = pointer.baseAddress {
                    if let results = self.module.predict(withBuffer: baseAddress) as? [Float], let resultValue = results.first {
                        self.printLevel(probability: resultValue)
                    }
                    
                }
            }
            
            DispatchQueue.main.async {
                // Updatethe latest samples for waveform
                self.latestSamples = Array(self.sampleBuffer)
            }
        }
        
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
            isStreaming = true
        } catch {
            print("Error starting the audio engine: \(error.localizedDescription)")
        }
        
    }
    
    
    func stopStreaming() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        isStreaming = false
    }
    
    func printLevel(probability: Float) {
        // Determine the number of blocks to display based on probability
        let numBlocks = Int(probability * 10)  // Using 10 blocks for full scale
        let blocks = String(repeating: "█", count: numBlocks)
        let spaces = String(repeating: " ", count: 10 - numBlocks)
        
        // Print the progress bar
        print("\r[\(blocks)\(spaces)] \(probability)", terminator: "")
        fflush(stdout)  // Force the print to display immediately
    }
    var recordingFormat: AVAudioFormat!

    func recordTwoSecondClip(completion: @escaping ([Float]) -> Void) {
        var recordedSamples: [Float] = []
        let requiredSamplesCount = Int(sampleRate * 2.0) // 2 seconds of samples
        
        let inputNode = audioEngine.inputNode
        self.recordingFormat = inputNode.outputFormat(forBus: 0)
        
        let tap: Void = inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(sampleRate * bufferTimeInterval), format: recordingFormat) { (buffer, time) in
            
            let dataPointer = buffer.floatChannelData![0]
            let arr = Array(UnsafeBufferPointer(start: dataPointer, count: Int(buffer.frameLength)))
            
            recordedSamples.append(contentsOf: arr)
            
            if recordedSamples.count >= requiredSamplesCount {
                // When we have captured 2 seconds of audio, remove the tap and stop the engine
                inputNode.removeTap(onBus: 0)
                self.audioEngine.stop()
                
                // Trim the array to have exactly 2 seconds of samples
                recordedSamples = Array(recordedSamples.prefix(requiredSamplesCount))
                
                // Call completion with the recorded samples
                completion(recordedSamples)
            }
        }
        
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
        } catch {
            print("Error starting the audio engine: \(error.localizedDescription)")
        }
    }
    
    
    
    func saveAudioToFile(samples: [Float]) {
        let buffer = AVAudioPCMBuffer(pcmFormat: recordingFormat, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channelMemory = buffer.floatChannelData![0]
        for i in 0..<samples.count {
            channelMemory[i] = samples[i]
        }
        
        let docsDir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
        let audioURL = docsDir.appendingPathComponent("audioClip.caf")
        
        guard let audioFile = try? AVAudioFile(forWriting: audioURL, settings: buffer.format.settings) else {
            print("Error: Could not create AVAudioFile")
            return
        }
        
        do {
            try audioFile.write(from: buffer)
            print("Saved audio to:", audioURL)
        } catch {
            print("Error saving audio:", error)
        }
    }
    
    
    
    
    
    
}
