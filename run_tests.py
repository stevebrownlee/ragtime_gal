#!/usr/bin/env python3
"""
Phase 4: Simple Test Runner Script

This script provides an easy way to run Phase 4 tests with various options
and configurations. It supports running individual test suites or the complete
test suite with customizable parameters.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_config import test_config, get_enabled_test_suites

def setup_logging(level: str = "INFO", detailed: bool = False):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if detailed:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('phase4_tests.log')
        ]
    )

def run_integration_tests() -> bool:
    """Run integration test suite."""
    try:
        from test_integration_complete import IntegrationTestSuite

        print("🔗 Running Integration Tests...")
        suite = IntegrationTestSuite()
        results = suite.run_integration_test_suite()
        return results.get("overall_success", False)
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False

def run_performance_tests() -> bool:
    """Run performance test suite."""
    try:
        from test_performance import PerformanceTestSuite

        print("⚡ Running Performance Tests...")
        suite = PerformanceTestSuite()
        results = suite.run_performance_test_suite()
        return results.get("overall_success", False)
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return False

def run_validation_tests() -> bool:
    """Run validation test suite."""
    try:
        from test_validation import ValidationTestSuite

        print("✅ Running Validation Tests...")
        suite = ValidationTestSuite()
        results = suite.run_validation_test_suite()
        return results.get("overall_success", False)
    except Exception as e:
        print(f"❌ Validation tests failed: {e}")
        return False

def run_regression_tests() -> bool:
    """Run regression test suite."""
    try:
        from test_regression import RegressionTestSuite

        print("🔄 Running Regression Tests...")
        suite = RegressionTestSuite()
        results = suite.run_regression_test_suite()
        return results.get("overall_success", False)
    except Exception as e:
        print(f"❌ Regression tests failed: {e}")
        return False

def run_complete_test_suite() -> bool:
    """Run the complete Phase 4 test suite."""
    try:
        from test_phase4_complete import Phase4TestRunner

        print("🚀 Running Complete Phase 4 Test Suite...")
        runner = Phase4TestRunner()
        results = runner.run_all_tests()
        return results.get("summary", {}).get("overall_success", False)
    except Exception as e:
        print(f"❌ Complete test suite failed: {e}")
        return False

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Phase 4 Test Runner - Comprehensive Testing and Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                          # Run complete test suite
  python run_tests.py --suite integration     # Run only integration tests
  python run_tests.py --suite performance     # Run only performance tests
  python run_tests.py --list-suites           # List available test suites
  python run_tests.py --quick                 # Run with reduced dataset sizes
  python run_tests.py --no-phase3             # Run without Phase 3 components
  python run_tests.py --verbose               # Enable detailed logging
        """
    )

    parser.add_argument(
        '--suite', '-s',
        choices=['integration', 'performance', 'validation', 'regression', 'complete'],
        help='Run specific test suite (default: complete)'
    )

    parser.add_argument(
        '--list-suites', '-l',
        action='store_true',
        help='List available test suites and exit'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run tests with reduced dataset sizes for faster execution'
    )

    parser.add_argument(
        '--no-phase3',
        action='store_true',
        help='Disable Phase 3 components for testing'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable detailed logging'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Performance scaling factor (default: 1.0)'
    )

    parser.add_argument(
        '--no-latency',
        action='store_true',
        help='Disable latency simulation in mock components'
    )

    parser.add_argument(
        '--simulate-errors',
        action='store_true',
        help='Enable error simulation in mock components'
    )

    args = parser.parse_args()

    # List available test suites
    if args.list_suites:
        print("Available Test Suites:")
        print("  integration  - End-to-end workflow and component interaction testing")
        print("  performance  - Load testing and performance benchmarking")
        print("  validation   - Quality assurance and effectiveness testing")
        print("  regression   - Backward compatibility and functionality preservation")
        print("  complete     - All test suites with comprehensive reporting")
        return

    # Configure environment based on arguments
    if args.no_phase3:
        os.environ['ENABLE_PHASE3'] = 'false'

    if args.quick:
        os.environ['TEST_PERFORMANCE_SCALE'] = '0.5'
    elif args.scale != 1.0:
        os.environ['TEST_PERFORMANCE_SCALE'] = str(args.scale)

    if args.no_latency:
        os.environ['SIMULATE_LATENCY'] = 'false'

    if args.simulate_errors:
        os.environ['SIMULATE_ERRORS'] = 'true'

    # Setup logging
    setup_logging(args.log_level, args.verbose)

    # Print configuration
    print("=" * 60)
    print("PHASE 4: TESTING AND VALIDATION SUITE")
    print("=" * 60)
    print(f"Phase 3 Enabled: {test_config.environment.enable_phase3}")
    print(f"Performance Scale: {test_config.environment.performance_scale_factor}")
    print(f"Simulate Latency: {test_config.environment.simulate_latency}")
    print(f"Simulate Errors: {test_config.environment.simulate_errors}")
    print(f"Log Level: {args.log_level}")
    print("")

    # Run selected test suite
    suite = args.suite or 'complete'
    success = False

    try:
        if suite == 'integration':
            success = run_integration_tests()
        elif suite == 'performance':
            success = run_performance_tests()
        elif suite == 'validation':
            success = run_validation_tests()
        elif suite == 'regression':
            success = run_regression_tests()
        elif suite == 'complete':
            success = run_complete_test_suite()
        else:
            print(f"❌ Unknown test suite: {suite}")
            return 1

        # Print final result
        print("\n" + "=" * 60)
        if success:
            print("🎉 ALL TESTS PASSED!")
            print("The feedback-driven embedding improvement system is ready for production.")
            return 0
        else:
            print("❌ SOME TESTS FAILED!")
            print("Please review the test results and address any issues.")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())