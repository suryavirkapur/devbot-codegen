#!/usr/bin/env python3
"""
Test runner for DevBot Python Backend

This script runs all tests and generates coverage reports.
Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, verbose=False):
    """Run a command and handle output"""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, check=True)
        if not verbose and result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run DevBot tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--unit", "-u", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", "-i", action="store_true", help="Run only integration tests")
    parser.add_argument("--html-cov", action="store_true", help="Generate HTML coverage report")
    
    args = parser.parse_args()
    
    # Set up base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.html_cov:
            cmd.append("--cov-report=html")
    
    # Filter tests
    if args.unit:
        cmd.append("tests/test_models.py")
        cmd.append("tests/test_services.py")
    elif args.integration:
        cmd.append("tests/test_api.py")
    else:
        cmd.append("tests/")
    
    print("🚀 Running DevBot Tests...")
    print("=" * 50)
    
    # Check if test dependencies are installed
    try:
        import pytest
        import pytest_asyncio
    except ImportError as e:
        print(f"❌ Missing test dependencies: {e}")
        print("💡 Install test dependencies with: pip install -r requirements-test.txt")
        return 1
    
    # Run the tests
    success = run_command(cmd, args.verbose)
    
    if success:
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        
        if args.coverage and args.html_cov:
            html_dir = Path("htmlcov")
            if html_dir.exists():
                print(f"📊 HTML coverage report generated: {html_dir / 'index.html'}")
        
        return 0
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 