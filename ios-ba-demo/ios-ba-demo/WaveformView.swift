//
//  WaveformView.swift
//  ios-ba-demo
//
//  Created by Ruben Nunez on 06.11.23.
//

import Foundation
import SwiftUI


struct WaveformView: View {
    var samples: [Float]
    private let waveformResolution = 500 // Number of lines in the waveform

    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let widthPerSample = geometry.size.width / CGFloat(waveformResolution)
                let centerY = geometry.size.height / 2
                
                for (index, sample) in resampled().enumerated() {
                    let x = CGFloat(index) * widthPerSample
                    let adjustedSample = (CGFloat(sample) + 1) / 2  
                    let y = centerY - adjustedSample * (geometry.size.height / 2)
                    
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(Color.blue, lineWidth: 2)
        }
    }
    
    private func resampled() -> [Float] {
        let binSize = samples.count / waveformResolution
        guard binSize > 0 else {
            return samples
        }
        return stride(from: 0, to: samples.count - binSize, by: binSize).map { index in
            Array(samples[index..<index+binSize]).average
        }
    }

}

extension Array where Element == Float {
    var average: Float {
        if isEmpty { return 0.0 }
        let sum = reduce(0, +)
        return sum / Float(count)
    }
}
