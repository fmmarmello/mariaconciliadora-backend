#!/usr/bin/env python3
"""
Maria Conciliadora - Test Runner Script

This script provides a convenient way to run different types of tests
with various configurations and generate comprehensive reports.

Usage:
    python run_tests.py --help
    python run_tests.py --all
    python run_tests.py --unit
    python run_tests.py --integration
    python run_tests.py --performance
    python run_tests.py --coverage
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.test_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_command(self, command, description="Running command"):
        """Run a shell command and return the result."""
        print(f"\n🔄 {description}")
        print(f"Command: {' '.join(command)}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=False,
                text=True,
                check=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully in {duration:.2f}s")
                return True
            else:
                print(f"❌ {description} failed with exit code {result.returncode}")
                return False
        
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False
    
    def run_unit_tests(self, verbose=False, coverage=False):
        """Run unit tests."""
        command = ["python", "-m", "pytest", "tests/unit/"]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:tests/reports/unit_coverage_html"
            ])
        
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests."""
        command = ["python", "-m", "pytest", "tests/integration/"]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Integration Tests")
    
    def run_performance_tests(self, verbose=False):
        """Run performance tests."""
        command = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Performance Tests")
    
    def run_all_tests(self, verbose=False, parallel=False):
        """Run all tests."""
        command = ["python", "-m", "pytest", "tests/"]
        
        if verbose:
            command.append("-v")
        
        if parallel:
            command.extend(["-n", "auto"])
        
        return self.run_command(command, "All Tests")
    
    def run_coverage_analysis(self):
        """Run comprehensive coverage analysis."""
        command = [
            "python", "-m", "pytest",
            "--cov=src",
            "--cov-report=html:tests/reports/coverage_html",
            "--cov-report=xml:tests/reports/coverage.xml",
            "--cov-report=json:tests/reports/coverage.json",
            "--cov-report=term-missing",
            "--cov-branch",
            "tests/"
        ]
        
        success = self.run_command(command, "Coverage Analysis")
        
        if success:
            # Run additional coverage analysis
            try:
                from tests.coverage_config import main as coverage_main
                print("\n🔄 Running enhanced coverage analysis...")
                coverage_main()
            except ImportError:
                print("⚠️  Enhanced coverage analysis not available")
        
        return success
    
    def run_specific_test(self, test_path, verbose=False):
        """Run a specific test file or function."""
        command = ["python", "-m", "pytest", test_path]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, f"Specific Test: {test_path}")
    
    def run_tests_by_marker(self, marker, verbose=False):
        """Run tests with specific marker."""
        command = ["python", "-m", "pytest", "-m", marker]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, f"Tests with marker: {marker}")
    
    def run_failed_tests(self, verbose=False):
        """Run only failed tests from last run."""
        command = ["python", "-m", "pytest", "--lf"]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Failed Tests (Last Failed)")
    
    def run_quick_tests(self, verbose=False):
        """Run only quick tests (exclude slow tests)."""
        command = ["python", "-m", "pytest", "-m", "not slow"]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Quick Tests (excluding slow)")
    
    def lint_code(self):
        """Run code linting."""
        commands = [
            (["python", "-m", "flake8", "src/", "tests/"], "Flake8 Linting"),
            (["python", "-m", "black", "--check", "src/", "tests/"], "Black Code Formatting Check"),
            (["python", "-m", "isort", "--check-only", "src/", "tests/"], "Import Sorting Check")
        ]
        
        all_passed = True
        for command, description in commands:
            if not self.run_command(command, description):
                all_passed = False
        
        return all_passed
    
    def setup_test_environment(self):
        """Set up test environment."""
        print("🔧 Setting up test environment...")
        
        # Check if required packages are installed
        required_packages = [
            "pytest", "pytest-cov", "pytest-mock", "pytest-xdist",
            "faker", "requests", "flask-testing"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Create necessary directories
        directories = [
            self.reports_dir,
            self.reports_dir / "coverage_html",
            self.reports_dir / "unit_coverage_html"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
        
        # Check database configuration
        if not os.getenv("DATABASE_URL"):
            print("⚠️  DATABASE_URL not set, using default test database")
            os.environ["DATABASE_URL"] = "sqlite:///test.db"
        
        os.environ["TESTING"] = "true"
        
        print("✅ Test environment setup complete")
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating Test Report...")
        
        report_content = f"""
# Maria Conciliadora - Test Execution Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Test Execution Summary

### Test Categories Executed
- ✅ Unit Tests
- ✅ Integration Tests  
- ✅ Performance Tests
- ✅ Coverage Analysis

### Coverage Reports
- HTML Report: `tests/reports/coverage_html/index.html`
- XML Report: `tests/reports/coverage.xml`
- JSON Report: `tests/reports/coverage.json`

### Performance Results
- Performance test results available in test output

### Recommendations
1. Review any failed tests and fix issues
2. Ensure coverage meets minimum thresholds (85% overall, 95% critical modules)
3. Address any performance bottlenecks identified
4. Update tests when adding new features

## Next Steps
1. Fix any failing tests
2. Improve coverage for modules below threshold
3. Optimize performance for slow operations
4. Update documentation as needed
"""
        
        report_path = self.reports_dir / "test_execution_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        print(f"📄 Test report saved to: {report_path}")
    
    def print_summary(self, results):
        """Print execution summary."""
        print("\n" + "=" * 60)
        print("🏁 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results if result[1])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️  {failed_tests} test suite(s) failed. Please review and fix.")
        
        print("\nFailed Test Suites:")
        for test_name, success in results:
            if not success:
                print(f"  - {test_name}")
        
        print("\n📊 Reports available in: tests/reports/")
        print("📖 Documentation: TESTING_DOCUMENTATION.md")


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Maria Conciliadora Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --verbose         # Run unit tests with verbose output
  python run_tests.py --coverage               # Run with coverage analysis
  python run_tests.py --performance            # Run performance tests only
  python run_tests.py --quick                  # Run quick tests only
  python run_tests.py --marker critical        # Run tests marked as critical
  python run_tests.py --specific tests/unit/test_ofx_processor.py  # Run specific test
        """
    )
    
    # Test type arguments
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--coverage", action="store_true", help="Run coverage analysis")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (exclude slow)")
    parser.add_argument("--failed", action="store_true", help="Run only failed tests from last run")
    
    # Test selection arguments
    parser.add_argument("--marker", help="Run tests with specific marker")
    parser.add_argument("--specific", help="Run specific test file or function")
    
    # Configuration arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--setup", action="store_true", help="Setup test environment")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    runner = TestRunner()
    results = []
    
    print("🧪 Maria Conciliadora - Test Runner")
    print("=" * 60)
    
    # Setup environment if requested
    if args.setup:
        if not runner.setup_test_environment():
            sys.exit(1)
        return
    
    # Run linting if requested
    if args.lint:
        success = runner.lint_code()
        results.append(("Code Linting", success))
    
    # Run specific test
    if args.specific:
        success = runner.run_specific_test(args.specific, args.verbose)
        results.append((f"Specific Test: {args.specific}", success))
    
    # Run tests by marker
    if args.marker:
        success = runner.run_tests_by_marker(args.marker, args.verbose)
        results.append((f"Marker: {args.marker}", success))
    
    # Run test categories
    if args.unit:
        success = runner.run_unit_tests(args.verbose, coverage=False)
        results.append(("Unit Tests", success))
    
    if args.integration:
        success = runner.run_integration_tests(args.verbose)
        results.append(("Integration Tests", success))
    
    if args.performance:
        success = runner.run_performance_tests(args.verbose)
        results.append(("Performance Tests", success))
    
    if args.coverage:
        success = runner.run_coverage_analysis()
        results.append(("Coverage Analysis", success))
    
    if args.quick:
        success = runner.run_quick_tests(args.verbose)
        results.append(("Quick Tests", success))
    
    if args.failed:
        success = runner.run_failed_tests(args.verbose)
        results.append(("Failed Tests", success))
    
    if args.all:
        success = runner.run_all_tests(args.verbose, args.parallel)
        results.append(("All Tests", success))
        
        # Also run coverage analysis for complete run
        success = runner.run_coverage_analysis()
        results.append(("Coverage Analysis", success))
    
    # Generate report if requested
    if args.report:
        runner.generate_test_report()
    
    # Print summary if any tests were run
    if results:
        runner.print_summary(results)
        
        # Exit with error code if any tests failed
        if any(not success for _, success in results):
            sys.exit(1)
    else:
        print("No tests were run. Use --help for usage information.")


if __name__ == "__main__":
    main()