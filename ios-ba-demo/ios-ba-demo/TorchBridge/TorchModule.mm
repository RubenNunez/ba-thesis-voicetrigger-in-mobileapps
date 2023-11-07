#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
 @protected
  torch::jit::mobile::Module _model;
    
@protected
 torch::jit::mobile::Module _transform;
}

- (nullable instancetype)initWithModelPath:(NSString*)modelPath andTransformPath:(NSString*)transformPath {
  self = [super init];
  if (self) {
    try {
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

    c10::InferenceMode guard;
      
    // NSLog(@"Input Tensor shape to transform: %s", tensorSizesToStr(tensor).c_str());

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
