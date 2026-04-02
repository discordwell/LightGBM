/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef LGBM_USE_METAL

// C++ standard headers first (before ObjC imports to avoid namespace conflicts)
#include "metal_utils.hpp"
#include "metal_leaf_splits.hpp"
#include "metal_best_split_finder.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <dispatch/dispatch.h>

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace LightGBM {

// ---------------------------------------------------------------------------
//  MetalDevice — singleton state
// ---------------------------------------------------------------------------

static id<MTLDevice>         g_device   = nil;
static id<MTLCommandQueue>   g_queue    = nil;
static id<MTLLibrary>        g_library  = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

static void InitDevice() {
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    g_device = MTLCreateSystemDefaultDevice();
    METAL_CHECK(g_device != nil, "No Metal device found");
    Log::Info("[Metal] Using device: %s",
              [[g_device name] UTF8String]);
  });
}

static void InitQueue() {
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    InitDevice();
    g_queue = [g_device newCommandQueue];
    METAL_CHECK(g_queue != nil, "Failed to create Metal command queue");
  });
}

static NSString* FindMetallibPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&MetalDevice::GetDevice), &info) && info.dli_fname) {
    NSString* dylib = [NSString stringWithUTF8String:info.dli_fname];
    NSString* dir = [dylib stringByDeletingLastPathComponent];
    NSString* candidate = [dir stringByAppendingPathComponent:@"lib_lightgbm.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
      return candidate;
    }
    candidate = [[dir stringByAppendingPathComponent:@"../lib"]
                      stringByAppendingPathComponent:@"lib_lightgbm.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
      return candidate;
    }
  }
  NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
  NSString* candidate = [cwd stringByAppendingPathComponent:@"lib_lightgbm.metallib"];
  if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
    return candidate;
  }
  return nil;
}

static void InitLibrary() {
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    InitDevice();
    NSString* path = FindMetallibPath();
    METAL_CHECK(path != nil, "Cannot locate lib_lightgbm.metallib");
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:path];
    g_library = [g_device newLibraryWithURL:url error:&error];
    METAL_CHECK(g_library != nil,
                [[error localizedDescription] UTF8String]);
  });
}

// ---------------------------------------------------------------------------
//  MetalDevice — public interface
// ---------------------------------------------------------------------------

void* MetalDevice::GetDevice() {
  InitDevice();
  return (__bridge void*)g_device;
}

void* MetalDevice::GetQueue() {
  InitQueue();
  return (__bridge void*)g_queue;
}

void* MetalDevice::GetLibrary() {
  InitLibrary();
  return (__bridge void*)g_library;
}

void* MetalDevice::GetPipeline(const char* function_name) {
  InitLibrary();
  std::string key(function_name);
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) {
    return (__bridge void*)it->second;
  }
  NSString* name = [NSString stringWithUTF8String:function_name];
  id<MTLFunction> func = [g_library newFunctionWithName:name];
  METAL_CHECK(func != nil, function_name);
  NSError* error = nil;
  id<MTLComputePipelineState> pso =
      [g_device newComputePipelineStateWithFunction:func error:&error];
  METAL_CHECK(pso != nil,
              [[error localizedDescription] UTF8String]);
  g_pipelines[key] = pso;
  return (__bridge void*)pso;
}

// ---------------------------------------------------------------------------
//  MetalBuffer<T> — implementation helpers
// ---------------------------------------------------------------------------

template <typename T>
T* MetalBuffer<T>::data() const {
  if (mtl_buffer_ == nullptr) {
    return nullptr;
  }
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)mtl_buffer_;
  return reinterpret_cast<T*>([buf contents]);
}

template <typename T>
void MetalBuffer<T>::Allocate(size_t n) {
  InitDevice();
  size_t bytes = n * sizeof(T);
  id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
  METAL_CHECK(buf != nil, "Metal buffer allocation failed");
  mtl_buffer_ = (__bridge_retained void*)buf;
  size_ = n;
}

template <typename T>
void MetalBuffer<T>::Release() {
  if (mtl_buffer_ != nullptr) {
    CFRelease(mtl_buffer_);
    mtl_buffer_ = nullptr;
  }
  size_ = 0;
}

// Explicit instantiations for all types used in the tree learner pipeline.
template class MetalBuffer<int8_t>;
template class MetalBuffer<uint8_t>;
template class MetalBuffer<int16_t>;
template class MetalBuffer<uint16_t>;
template class MetalBuffer<int32_t>;
template class MetalBuffer<int64_t>;
template class MetalBuffer<uint32_t>;
template class MetalBuffer<uint64_t>;
template class MetalBuffer<float>;
template class MetalBuffer<double>;
template class MetalBuffer<MetalLeafSplitsStruct>;
template class MetalBuffer<MetalSplitResult>;
template class MetalBuffer<MetalSplitFindTask>;

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
