//
//  AudioInputManager.swift
//  ios-ba-demo
//
//  Created by Ruben Nunez on 08.11.23.
//

import Foundation
import UIKit
import AVFoundation
import SwiftUI
import Combine

struct IdentifiableURL: Identifiable {
    let id: UUID = UUID() // This provides a unique identifier
    let url: URL
}

class AudioDataManager: ObservableObject, AudioInputManagerDelegate {
    
    @Published var lastCapturedData: [Float32] = Array(repeating: 0.0, count: Int(32000))
    @Published var isStreaming: Bool = false
    @Published var fileURL: IdentifiableURL?
    
    private var audioInputManager: AudioInputManager?
    
    private lazy var module: TorchModule = {
        if let modelFilePath = Bundle.main.path(forResource: "model", ofType: "ptl"),
           let transformFilePath = Bundle.main.path(forResource: "transform", ofType: "ptl"),
           let wav2vecFilePat = Bundle.main.path(forResource: "wav2vec2", ofType: "ptl"),
           let module = TorchModule(modelPath: modelFilePath,
                                    andTransformPath: transformFilePath,
                                    andWav2VecFromPath: wav2vecFilePat) {
            return module
        } else {
            fatalError("Can't find the model or transform file!")
        }
    }()
    
    init() {
        self.audioInputManager = AudioInputManager(sampleRate: 16000)
        self.audioInputManager?.delegate = self
    }
    
    func startAudioCapture() {
        audioInputManager?.checkPermissionsAndStartTappingMicrophone()
        isStreaming = true
    }
    
    func stopAudioCapture() {
        audioInputManager?.stopTappingMicrophone()
        isStreaming = false
        
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let outputFileURL = paths[0].appendingPathComponent("output.wav")
        
        DispatchQueue.main.async {
            self.fileURL = IdentifiableURL(url: outputFileURL)
        }
    }
    
    func audioInputManagerDidFailToAchievePermission(_ audioInputManager: AudioInputManager) {
        DispatchQueue.main.async {
            print("Permission denied by the user")
        }
    }
    
    func audioInputManager(_ audioInputManager: AudioInputManager, didCaptureChannelData: [Float]) {
        DispatchQueue.main.async {
            
            self.lastCapturedData = Array(self.lastCapturedData.dropFirst(didCaptureChannelData.count))
            self.lastCapturedData.append(contentsOf: didCaptureChannelData)
            
            assert(self.lastCapturedData.count == 32000)
            
            // Pass the buffer for inference
            self.lastCapturedData.withUnsafeMutableBufferPointer { pointer in
                if let baseAddress = pointer.baseAddress {
                    if let results = self.module.predict(withBuffer: baseAddress) as? [Float], let resultValue = results.first {
                        var text = ""
                        if(resultValue > 0.5){
                            //text = self.module.recognize(baseAddress, bufferLength: 32000)
                        }
                        
                        self.printLevel(probability: resultValue, text: text)
                    }
                }
            }
        }
    }
    
    func printLevel(probability: Float, text: String) {
        let numBlocks = Int(probability * 10)
        let blocks = String(repeating: "█", count: numBlocks)
        let spaces = String(repeating: " ", count: 10 - numBlocks)
        
        let formattedNumber = String(format: "%.3f", probability)
        
        print("\r[\(blocks)\(spaces)] \(formattedNumber) -> \(text)", terminator: "")
    }
}


public protocol AudioInputManagerDelegate: AnyObject {
    func audioInputManagerDidFailToAchievePermission(_ audioInputManager: AudioInputManager)
    func audioInputManager(_ audioInputManager: AudioInputManager, didCaptureChannelData: [Float32])
}

public class AudioInputManager {
    // MARK: - Constants
    public let bufferSize: Int
    
    private let sampleRate: Int
    private let conversionQueue = DispatchQueue(label: "conversionQueue")
    private let bufferTimeInterval: Double = 0.1
    
    // MARK: - Variables
    public weak var delegate: AudioInputManagerDelegate?
    
    private var audioEngine = AVAudioEngine()
    private var outputFile: AVAudioFile?
    
    // MARK: - Methods
    
    public init(sampleRate: Int) {
        self.sampleRate = sampleRate
        self.bufferSize = sampleRate * 2
    }
    
    public func checkPermissionsAndStartTappingMicrophone() {
        switch AVAudioSession.sharedInstance().recordPermission {
        case .granted:
            startTappingMicrophone()
        case .denied:
            delegate?.audioInputManagerDidFailToAchievePermission(self)
        case .undetermined:
            requestPermissions()
        @unknown default:
            fatalError()
        }
    }
    
    public func requestPermissions() {
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            if granted {
                self.startTappingMicrophone()
            } else {
                self.checkPermissionsAndStartTappingMicrophone()
            }
        }
    }
    
    /// Starts tapping the microphone input and converts it into the format for which the model is trained and
    /// periodically returns it in the block
    public func startTappingMicrophone() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord)
            try audioSession.setPreferredSampleRate(Double(sampleRate))
            // try audioSession.setPreferredIOBufferDuration(bufferTimeInterval)
            try audioSession.setMode(.default)
            // if audioSession.isInputGainSettable {try audioSession.setInputGain(1)}
            /*if let desc = audioSession.availableInputs?.first(where: { (desc) -> Bool in
             return desc.portType == AVAudioSession.Port.builtInMic || desc.portType == AVAudioSession.Port.bluetoothA2DP
             }){
             do{
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
        
        
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        
        guard let recordingFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ), let formatConverter = AVAudioConverter(from:inputFormat, to: recordingFormat) else { return }
        
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let fileURL = paths[0].appendingPathComponent("output.wav")
        
        
        do {
            self.outputFile = try AVAudioFile(forWriting: fileURL, settings: recordingFormat.settings)
        } catch {
            print(error.localizedDescription)
            return
        }
        
        // installs a tap on the audio engine and specifying the buffer size and the input format.
        inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(recordingFormat.sampleRate * bufferTimeInterval), format: inputFormat) {
            buffer, _ in
            
            self.conversionQueue.async { [self] in
                
                // An AVAudioConverter is used to convert the microphone input to the format required
                guard let pcmBuffer = AVAudioPCMBuffer(
                    pcmFormat: recordingFormat,
                    frameCapacity: AVAudioFrameCount(recordingFormat.sampleRate * bufferTimeInterval)
                ) else { return }
                
                let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                    outStatus.pointee = AVAudioConverterInputStatus.haveData
                    return buffer
                }
                
                var error: NSError?
                formatConverter.convert(to: pcmBuffer, error: &error, withInputFrom: inputBlock)
                
                // Write buffer to file
                do {
                    try self.outputFile?.write(from: pcmBuffer)
                } catch {
                    print(error.localizedDescription)
                }
                
                if let error = error {
                    print(error.localizedDescription)
                    return
                }
                if let channelData = pcmBuffer.floatChannelData {
                    let channelDataValue = channelData.pointee
                    let channelDataValueArray = stride(
                        from: 0,
                        to: Int(pcmBuffer.frameLength),
                        by: buffer.stride
                    ).map { channelDataValue[$0] }
                    
                    self.delegate?.audioInputManager(self, didCaptureChannelData: channelDataValueArray)
                }
            }
        }
        
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func stopTappingMicrophone() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
    }
    
}
