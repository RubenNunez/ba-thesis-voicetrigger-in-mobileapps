#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

// Updated initializer to take two paths
- (nullable instancetype)initWithModelPath:(NSString*)modelPath
                          andTransformPath:(NSString*)transformPath
                        andWav2VecFromPath: (NSString*)wav2vecPath;

- (NSArray<NSNumber*>*)predictWithBuffer:(void*)buffer;

- (NSString*)recognize:(void*)wavBuffer bufferLength:(int)bufferLength;

@end

NS_ASSUME_NONNULL_END
