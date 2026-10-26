#!/usr/bin/env bash
# Runs the native XCTests (ios/Tests) on a connected iPhone, where Vision uses
# the Neural Engine and Apple Intelligence is available:
#
#   scripts/test-ios-device.sh <device-udid> <team-id>   (from the repo root)
#
# `xcrun xctrace list devices` prints the UDID. SwiftPM cannot run tests on a
# device, so this runs the podspec's test spec from the example app's Pods
# project, which hosts the bundle in a generated app. example/ios is generated
# by prebuild; the Podfile edit below is reapplied on every run.
set -euo pipefail

udid="${1:?usage: $0 <device-udid> <team-id>}"
team="${2:?usage: $0 <device-udid> <team-id>}"

cd "$(dirname "$0")/../example"
[ -d ios ] || npx expo prebuild --platform ios --no-install

# Autolinking adds the pod without its test spec; declare it ourselves.
sed -i '' 's|^  use_expo_modules!$|  use_expo_modules!(exclude: ["@insidepics/expo-apple-intelligence"])\
  pod "ExpoAppleIntelligence", :path => "../../ios", :testspecs => ["ExpoAppleIntelligenceTests"]|' ios/Podfile
(cd ios && pod install)

# Pods are built unsigned; a bundle loaded on a device must be signed.
xcodebuild test \
  -workspace ios/expoappleintelligenceexample.xcworkspace \
  -scheme ExpoAppleIntelligence-Unit-ExpoAppleIntelligenceTests \
  -destination "id=$udid" \
  -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$team" \
  CODE_SIGN_STYLE=Automatic \
  CODE_SIGNING_ALLOWED=YES \
  CODE_SIGN_IDENTITY="Apple Development"
