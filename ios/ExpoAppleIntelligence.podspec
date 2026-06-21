require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'ExpoAppleIntelligence'
  s.version        = package['version']
  s.summary        = package['description']
  s.description    = package['description']
  s.license        = package['license']
  s.author         = package['author']
  s.homepage       = package['homepage']
  s.platforms      = {
    :ios => '15.1',
    :osx => '13.0'
  }
  s.swift_version  = '5.9'
  s.source         = { :git => 'https://github.com/InsidePics/expo-apple-intelligence.git', :tag => "#{s.version}" }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.frameworks = 'Vision'
  s.weak_frameworks = 'FoundationModels', 'Speech', 'AVFAudio', 'ImagePlayground'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
  s.exclude_files = "Tests/**"

  s.test_spec 'ExpoAppleIntelligenceTests' do |test_spec|
    test_spec.source_files = 'Tests/**/*.swift'
    test_spec.resources = 'Tests/Fixtures/*'
    test_spec.frameworks = 'XCTest'
  end
end
