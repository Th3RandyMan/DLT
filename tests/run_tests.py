#!/usr/bin/env python3
"""
Test runner script for DLT Framework

This script provides convenient ways to run the test suite with different configurations.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --quick            # Run quick tests only (no slow/gpu tests)
    python run_tests.py --integration      # Run integration tests
    python run_tests.py --unit             # Run unit tests only
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --torch-only       # Run only PyTorch-related tests
    python run_tests.py --sklearn-only     # Run only sklearn-related tests
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(command, description, project_root):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, cwd=project_root)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run DLT Framework tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick tests only (skip slow and GPU tests)')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests only (exclude integration)')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--torch-only', action='store_true',
                       help='Run only PyTorch-related tests')
    parser.add_argument('--sklearn-only', action='store_true',
                       help='Run only sklearn-related tests (no torch marker)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--parallel', '-n', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--file', '-f', type=str,
                       help='Run specific test file')
    parser.add_argument('--test', '-k', type=str,
                       help='Run specific test by name pattern')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    if not (project_root / 'tests').exists():
        print("❌ Tests directory not found. Please run from project root or tests directory.")
        sys.exit(1)
    
    # Change to project root directory
    import os
    os.chdir(project_root)
    
    # Check if dependencies are installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Please install development dependencies:")
        print("   uv sync --group dev")
        sys.exit(1)
    
    # Build base command
    cmd = ['python', '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    # Add parallel execution
    if args.parallel > 1:
        cmd.extend(['-n', str(args.parallel)])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term-missing'])
    
    # Add test selection based on arguments
    if args.quick:
        cmd.extend(['-m', 'not slow and not gpu'])
        description = "Running quick tests (excluding slow and GPU tests)"
    elif args.integration:
        cmd.extend(['-m', 'integration'])
        description = "Running integration tests"
    elif args.unit:
        cmd.extend(['-m', 'not integration'])
        description = "Running unit tests"
    elif args.torch_only:
        cmd.extend(['-m', 'torch'])
        description = "Running PyTorch-related tests"
    elif args.sklearn_only:
        cmd.extend(['-m', 'not torch'])
        description = "Running sklearn-related tests"
    else:
        description = "Running all tests"
    
    # Add specific file or test pattern
    if args.file:
        cmd.append(f'tests/{args.file}')
        description += f" from {args.file}"
    
    if args.test:
        cmd.extend(['-k', args.test])
        description += f" matching '{args.test}'"
    
    # Add test directory if no specific file
    if not args.file:
        cmd.append('tests/')
    
    # Run the tests
    print("🧪 DLT Framework Test Runner")
    print("=" * 50)
    
    success = run_command(cmd, description, project_root)
    
    if success:
        print("\n🎉 All tests completed successfully!")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("   - Terminal: see above")
            print("   - HTML: open htmlcov/index.html")
        
        print("\n💡 Tips:")
        print("   - Use --quick for faster development testing")
        print("   - Use --integration to test end-to-end workflows")
        print("   - Use --coverage to check test coverage")
        print("   - Use -f <filename> to run specific test files")
        print("   - Use -k <pattern> to run specific tests")
        
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        print("\n🔧 Debugging tips:")
        print("   - Check that all dependencies are installed: uv sync --group dev")
        print("   - Ensure PyTorch/TensorFlow are available for related tests")
        print("   - Run with --verbose for more detailed output")
        print("   - Run specific failing tests with -f <file> or -k <test_name>")
        sys.exit(1)


if __name__ == "__main__":
    main()