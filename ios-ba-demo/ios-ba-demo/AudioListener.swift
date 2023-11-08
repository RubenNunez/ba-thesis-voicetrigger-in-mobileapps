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
    private let sampleRate: Int = 16000
    private let bufferDuration: Double = 2.0
    private let bufferTimeInterval: Double = 0.1
    private let requiredSamplesCount: Int = 32000
    
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
        self.sampleBuffer = Array(repeating: 0.0, count: Int(requiredSamplesCount))
        
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record)
            try audioSession.setPreferredSampleRate(Double(sampleRate))
            // try audioSession.setPreferredIOBufferDuration(bufferTimeInterval)
            try audioSession.setMode(.default)
            /*if let desc = audioSession.availableInputs?.first(where: { (desc) -> Bool in
                return desc.portType == AVAudioSession.Port.builtInMic || desc.portType == AVAudioSession.Port.bluetoothA2DP
            }){
                    try audioSession.setPreferredInput(desc)
                } catch let error{
                    print(error)
                }
            }*/
            try audioSession.setActive(true)
        } catch {                   
            print("Error configuring the audio session: \(error.localizedDescription)")
            return // Early return if audio session setup fails
        }
        
        /*let settings = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ] as [String : Any]*/
        
        
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        print("Current input node sample rate: \(inputFormat.sampleRate)")
        
        //let recordingFormat = AVAudioFormat(settings: settings)
        let recordingFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: true)
        // assert(inputFormat.sampleRate == 16000)
        assert(recordingFormat?.sampleRate == 16000)
        
        inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(recordingFormat!.sampleRate * bufferTimeInterval), format: recordingFormat) { [weak self] (buffer, when) in
            guard let self = self else { return }
            
            let dataPointer = buffer.floatChannelData![0]
            let arr = Array(UnsafeBufferPointer(start: dataPointer, count: Int(buffer.frameLength)))

            // Roll the sample buffer to remove old audio and append new audio
            self.sampleBuffer = Array(self.sampleBuffer.dropFirst(arr.count))
            self.sampleBuffer.append(contentsOf: arr)
            
            assert(self.sampleBuffer.count == 32000)

            // Pass the buffer for inference
            self.sampleBuffer.withUnsafeMutableBufferPointer { pointer in
                if let baseAddress = pointer.baseAddress {
                    if let results = self.module.predict(withBuffer: baseAddress) as? [Float], let resultValue = results.first {
                        self.printLevel(probability: resultValue)
                    }
                }
            }
            
            DispatchQueue.main.async {
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
        let numBlocks = Int(probability * 10)
        let blocks = String(repeating: "█", count: numBlocks)
        let spaces = String(repeating: " ", count: 10 - numBlocks)
        
        print("\r[\(blocks)\(spaces)] \(probability)", terminator: "")
        fflush(stdout)  // Force the print to display immediately
    }
    
}
