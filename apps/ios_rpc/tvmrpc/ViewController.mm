/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ViewController.mm
 */

#import "ViewController.h"
#include <string>

@implementation ViewController

enum TrackerCode {
  PUT = 3,
  UPDATE_INFO = 5,
  GET_PENDING_MATCHKEYS = 7,
  SUCCESS = 0
};

std::string generateUpdateInfoJson(std::string key) {
  return "[" + std::to_string(TrackerCode::UPDATE_INFO) + ", {\"key\": \"" + key + "\"}]";
}
std::string generatePutInfoJson(std::string key, int serverPort, std::string matchKey) {
  return "[" + std::to_string(TrackerCode::PUT) + ", \"" + key + "\", [" + std::to_string(serverPort) + ", "
              + "\"" + matchKey +  "\"], null]";
}

- (void)stream:(NSStream*)strm handleEvent:(NSStreamEvent)event {
  std::string buffer;
  switch (event) {
    case NSStreamEventOpenCompleted: {
      self.statusLabel.text = @"Connected";
      break;
    }
    case NSStreamEventHasBytesAvailable:
      if (strm == inputStream_) {
        [self onReadAvailable];
      }
      break;
    case NSStreamEventHasSpaceAvailable: {
      if (strm == outputStream_) {
        [self onWriteAvailable];
      }
      break;
    }
    case NSStreamEventErrorOccurred: {
      NSLog(@"%@", [strm streamError].localizedDescription);
      break;
    }
    case NSStreamEventEndEncountered: {
      [self close];
      // auto reconnect when normal end.
      [self open];
      break;
    }
    default: {
      NSLog(@"Unknown event");
    }
  }
}

- (void)onReadAvailable {
  NSLog(@"onReadAvailable");
  // Magic header for RPC data plane
  constexpr int kRPCMagic = 0xff271;
  // magic header for RPC tracker(control plane)
  constexpr int kRPCTrackerMagic = 0x2f271;
  // sucess response
  constexpr int kRPCSuccess = kRPCMagic + 0;
  // cannot found matched key in server
  constexpr int kRPCMismatch = kRPCMagic + 2;
  constexpr int kBufferSize = 4 << 10;
  if (!initialized_/* || registering_*/) {
    registering_ = false;
    int code;
    size_t nbytes = [inputStream_ read:reinterpret_cast<uint8_t*>(&code) maxLength:sizeof(code)];
    if (nbytes != sizeof(code)) {
      self.infoText.text = @"Fail to receive remote confirmation code.";
      [self close];
    } else if (code == kRPCMismatch) {
      self.infoText.text = @"Proxy server cannot find client that matches the key";
      [self close];
    } else if (code == kRPCMagic + 1) {
      self.infoText.text = @"Proxy server already have another server with same key";
      [self close];
    //} else if (code != kRPCMagic) {
    } else if (code != kRPCTrackerMagic) {
      self.infoText.text = @"Given address is not a TVM RPC Proxy";
      [self close];
    } else {
      initialized_ = true;
      self.statusLabel.text = @"Proxy connected.";
      ICHECK(handler_ != nullptr);
    }
  /*} else if (initialized_ && !registered_) {
    int recvCode;
    size_t nbytes = [inputStream_ read:reinterpret_cast<uint8_t*>(&recvCode) maxLength:sizeof(recvCode)];
    if (nbytes != sizeof(recvCode)) {
      self.infoText.text = @"Fail to receive remote confirmation code.";
      [self close];
    } else if (recvCode != kRPCSuccess) {
      self.infoText.text = @"Cannot register app in rpc_tracker";
      [self close];
    } else {
      registered_ = true;
    }*/
  } else if (initialized_/* && registered_*/) {
    while ([inputStream_ hasBytesAvailable]) {
      recvBuffer_.resize(kBufferSize);
      uint8_t* bptr = reinterpret_cast<uint8_t*>(&recvBuffer_[0]);
      size_t nbytes = [inputStream_ read:bptr maxLength:kBufferSize];
      recvBuffer_.resize(nbytes);
      int flag = 1;
      if ([outputStream_ hasSpaceAvailable]) {
        flag |= 2;
      }
      // always try to write
      try {
        flag = handler_(recvBuffer_, flag);
        if (flag == 2) {
          [self onShutdownReceived];
        }
      } catch (const dmlc::Error& e) {
        [self close];
      }
    }
  }
}

- (void)onShutdownReceived {
  [self close];
}

- (void)onWriteAvailable {
  NSLog(@"onWriteAvailable");
  if (initSendPtr_ < initBytes_.length()) {
    initSendPtr_ += [outputStream_ write:reinterpret_cast<uint8_t*>(&initBytes_[initSendPtr_])
                               maxLength:(initBytes_.length() - initSendPtr_)];
  } else if (!initialized_/* && !registered_*/) {
    std::ostringstream os;
    std::string key = generateUpdateInfoJson("server:" + key_);
    int keylen = static_cast<int>(key.length());
    os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
    os.write(key.c_str(), key.length());
    std::string keyBytes = os.str();
    [outputStream_ write:reinterpret_cast<uint8_t*>(&keyBytes[0])
               maxLength:(keyBytes.length())];
  } else if (!registered_ && !registering_) {
    std::ostringstream os;
    constexpr int serverPort = 5001;
    std::string key = generatePutInfoJson(key_, serverPort, matchKey_);
    int keylen = static_cast<int>(key.length());
    os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
    os.write(key.c_str(), key.length());
    std::string keyBytes = os.str();
    [outputStream_ write:reinterpret_cast<uint8_t*>(&keyBytes[0])
               maxLength:(keyBytes.length())];
    registering_ = true;
  } else {
    try {
      std::string dummy;
      int flag = handler_(dummy, 2);
      if (flag == 2) {
        [self onShutdownReceived];
      }
    } catch (const dmlc::Error& e) {
      [self close];
    }
  }
}

- (void)open {
  // Magic header for RPC data plane
  constexpr int kRPCMagic = 0xff271;
  // magic header for RPC tracker(control plane)
  constexpr int kRPCTrackerMagic = 0x2f271;
  // sucess response
  constexpr int kRPCSuccess = kRPCMagic + 0;
  // cannot found matched key in server
  constexpr int kRPCMismatch = kRPCMagic + 2;

  NSLog(@"Connecting to the proxy server..");
  // Initialize the data states.
  key_ = [self.proxyKey.text UTF8String];
  //key_ = "server:" + key_; // Add randomize key
  matchKey_ = key_ + ":" + std::to_string(((double)arc4random() / UINT32_MAX));
  std::ostringstream os;
  int rpc_magic = kRPCMagic;
  rpc_magic = kRPCTrackerMagic;
  os.write(reinterpret_cast<char*>(&rpc_magic), sizeof(rpc_magic));
  //int keylen = static_cast<int>(key_.length());
  //os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
  //os.write(key_.c_str(), key_.length());
  initialized_ = false;
  registered_ = false;
  registering_ = false;
  initBytes_ = os.str();
  initSendPtr_ = 0;
  // Initialize the network.
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  port_ = [self.proxyPort.text intValue];
  CFStreamCreatePairWithSocketToHost(NULL, (__bridge CFStringRef)self.proxyURL.text,
                                     port_, &readStream, &writeStream);
  inputStream_ = (__bridge_transfer NSInputStream*)readStream;
  outputStream_ = (__bridge_transfer NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];
  handler_ = tvm::runtime::CreateServerEventHandler(outputStream_, key_, "%toinit");
  ICHECK(handler_ != nullptr);
  self.infoText.text = @"";
  self.statusLabel.text = @"Connecting...";
}

- (void)close {
  NSLog(@"Closing the streams.");
  [inputStream_ close];
  [outputStream_ close];
  [inputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [inputStream_ setDelegate:nil];
  [outputStream_ setDelegate:nil];
  inputStream_ = nil;
  outputStream_ = nil;
  handler_ = nullptr;
  self.statusLabel.text = @"Disconnected";
}

- (IBAction)connect:(id)sender {
  [self open];
  [[self view] endEditing:YES];
}

- (IBAction)disconnect:(id)sender {
  [self close];
}

@end
