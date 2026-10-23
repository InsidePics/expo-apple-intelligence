// swift-tools-version:5.9
// Runs the native Vision tests without the Expo / React Native host app:
//   swift test                                              (macOS 26, AppKit)
//   xcodebuild test -scheme ExpoAppleIntelligence \
//     -destination 'platform=macOS,variant=Mac Catalyst'    (UIKit, as on iOS)
// The Expo module class needs ExpoModulesCore, so it is left out; every helper
// it calls is built and tested here. Not part of the npm package.
import PackageDescription

let package = Package(
  name: "ExpoAppleIntelligence",
  platforms: [.iOS("16.4"), .macOS("26.0")],
  products: [
    .library(name: "ExpoAppleIntelligence", targets: ["ExpoAppleIntelligence"])
  ],
  targets: [
    .target(
      name: "ExpoAppleIntelligence",
      path: "ios",
      exclude: [
        "ExpoAppleIntelligenceModule.swift",
        "ExpoAppleIntelligence.podspec",
        "Tests",
      ]
    ),
    .testTarget(
      name: "ExpoAppleIntelligenceTests",
      dependencies: ["ExpoAppleIntelligence"],
      path: "ios/Tests",
      resources: [.copy("Fixtures")]
    ),
  ]
)
