"""
Master test runner with coverage reporting
Runs all module tests and aggregates coverage metrics
Target: 80%+ overall coverage
"""

from coverage import CoverageTracker
from test_vector_ops import run_all_tests as test_vector_ops
from test_batch_processor import run_all_tests as test_batch_processor
from test_quality_metrics import run_all_tests as test_quality_metrics


fn main() raises:
    print("\n" + "=" * 60)
    print("🧪 VECTRO TEST SUITE - COMPREHENSIVE COVERAGE REPORT")
    print("=" * 60)
    
    var tracker = CoverageTracker()
    
    # Run all module tests and collect coverage
    print("\n📦 Running test suites...")
    
    var coverage_vector_ops = test_vector_ops()
    tracker.add_module(coverage_vector_ops)
    
    var coverage_batch_processor = test_batch_processor()
    tracker.add_module(coverage_batch_processor)
    
    var coverage_quality_metrics = test_quality_metrics()
    tracker.add_module(coverage_quality_metrics)
    
    # TODO: Add remaining module tests when created
    # var coverage_compression = test_compression_profiles()
    # tracker.add_module(coverage_compression)
    #
    # var coverage_storage = test_storage_mojo()
    # tracker.add_module(coverage_storage)
    #
    # var coverage_benchmark = test_benchmark_mojo()
    # tracker.add_module(coverage_benchmark)
    #
    # var coverage_streaming = test_streaming_quantizer()
    # tracker.add_module(coverage_streaming)
    #
    # var coverage_api = test_vectro_api()
    # tracker.add_module(coverage_api)
    #
    # var coverage_quantizer = test_quantizer()
    # tracker.add_module(coverage_quantizer)
    #
    # var coverage_standalone = test_vectro_standalone()
    # tracker.add_module(coverage_standalone)
    
    # Print final coverage report
    print("\n" + "=" * 60)
    print("📊 FINAL COVERAGE REPORT")
    print("=" * 60)
    tracker.print_summary()
    
    # Check if we met the 80% threshold
    print("\n" + "=" * 60)
    var threshold: Float32 = 80.0
    var passed = tracker.meets_threshold(threshold)
    
    if passed:
        print("✅ SUCCESS: Coverage meets", threshold, "% threshold!")
        print("=" * 60)
    else:
        print("❌ FAILED: Coverage below", threshold, "% threshold")
        print("   Please add more tests to improve coverage")
        print("=" * 60)
