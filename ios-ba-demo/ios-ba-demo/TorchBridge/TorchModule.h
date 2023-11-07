#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

// Updated initializer to take two paths
- (nullable instancetype)initWithModelPath:(NSString*)modelPath andTransformPath:(NSString*)transformPath;
- (NSArray<NSNumber*>*)predictWithBuffer:(void*)buffer;

@end

NS_ASSUME_NONNULL_END
