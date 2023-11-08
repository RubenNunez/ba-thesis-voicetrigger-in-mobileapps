#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
@protected
    torch::jit::mobile::Module _model;
    
@protected
    torch::jit::mobile::Module _transform;
    
@protected
    torch::jit::mobile::Module _wav2vec;
}

- (nullable instancetype)initWithModelPath:(NSString*)modelPath andTransformPath:(NSString*)transformPath andWav2VecFromPath: (NSString*)wav2vecPath{
    self = [super init];
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            _wav2vec = torch::jit::_load_for_mobile(wav2vecPath.UTF8String);
            _model = torch::jit::_load_for_mobile(modelPath.UTF8String);
            _transform = torch::jit::_load_for_mobile(transformPath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}
- (NSArray<NSNumber*>*)predictWithBuffer:(void*)buffer {
    try {
        at::Tensor tensor = torch::from_blob(buffer, {1, 32000}, at::kFloat);  // assuming 2 seconds * 16000 samples/sec
        //NSLog(@"Input Tensor shape to transform: %s", tensorSizesToStr(tensor).c_str());
        
        c10::InferenceMode guard;
        
        // Transform the raw audio first
        auto transformedTensor = _transform.forward({tensor}).toTensor();
        
        // Feed the transformed data to the main model
        auto logits = _model.forward({transformedTensor}).toTensor();
        
        // Apply sigmoid activation
        auto outputTensor = torch::sigmoid(logits);
        
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < outputTensor.size(0); i++) {
            [results addObject:@(floatBuffer[i])];
        }
        
        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

- (NSString*)recognize:(void*)wavBuffer bufferLength:(int)bufferLength{
    try {
        at::Tensor tensorInputs = torch::from_blob((void*)wavBuffer, {1, bufferLength}, at::kFloat);
        
        float* floatInput = tensorInputs.data_ptr<float>();
        if (!floatInput) {
            return nil;
        }
        NSMutableArray* inputs = [[NSMutableArray alloc] init];
        for (int i = 0; i < bufferLength; i++) {
            [inputs addObject:@(floatInput[i])];
        }
        
        c10::InferenceMode guard;
        
        //CFTimeInterval startTime = CACurrentMediaTime();
        auto result = _wav2vec.forward({ tensorInputs }).toStringRef();
        //CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        //NSLog(@"inference time:%f", elapsedTime);
            
        return [NSString stringWithCString:result.c_str() encoding:[NSString defaultCStringEncoding]];
    }
    catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

std::string tensorSizesToStr(const at::Tensor& tensor) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < tensor.dim(); ++i) {
        oss << tensor.size(i);
        if (i != tensor.dim() - 1) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

@end
