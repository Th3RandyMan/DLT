# DLT Framework Test Suite

This directory contains the comprehensive test suite for the DLT (Deep Learning Toolkit) framework. The tests ensure reliability, correctness, and maintainability of all framework components.

## 📁 Test Structure

```
tests/
├── README.md                    # This file - test documentation
├── conftest.py                  # Pytest configuration and shared fixtures
├── run_tests.py                 # Test runner script with various options
├── validate_framework.py        # Quick smoke tests and validation
├── test_config.py              # Configuration system tests
├── test_model.py               # Model creation and management tests
├── test_pipeline.py            # Data pipeline tests
├── test_trainer.py             # Training system tests
├── test_utils.py               # Utility functions tests
└── test_integration.py         # End-to-end integration tests
```

## 🧪 Test Categories

### Unit Tests
Individual component testing to ensure each module works correctly in isolation:

- **`test_config.py`** - Configuration system validation
- **`test_model.py`** - Model factory and management
- **`test_pipeline.py`** - Data processing pipelines
- **`test_trainer.py`** - Training algorithms and optimization
- **`test_utils.py`** - Helper functions and utilities

### Integration Tests
End-to-end workflow testing in **`test_integration.py`**:

- Complete training workflows across different frameworks
- Multi-framework model comparisons
- Pipeline integration with real datasets
- Configuration-driven experiments

### Validation Tests
Quick smoke tests in **`validate_framework.py`**:

- Import verification
- Basic functionality checks
- Framework availability testing
- Quick health checks

## 🏃‍♂️ Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py

# Run specific test files
pytest tests/test_config.py -v
pytest tests/test_model.py -v
```

### Test Runner Options

The `run_tests.py` script provides multiple testing modes:

```bash
# Quick tests (no slow/GPU tests)
python tests/run_tests.py --quick

# Integration tests only
python tests/run_tests.py --integration

# Unit tests only
python tests/run_tests.py --unit

# Run with coverage report
python tests/run_tests.py --coverage

# Framework-specific tests
python tests/run_tests.py --torch-only
python tests/run_tests.py --sklearn-only

# Verbose output
python tests/run_tests.py --verbose

# Run specific test patterns
python tests/run_tests.py --pattern "*config*"
```

### Direct Pytest Usage

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_model.py::TestDLTModel -v

# Run tests matching pattern
pytest tests/ -k "test_config" -v

# Run with coverage
pytest tests/ --cov=src/dlt --cov-report=html

# Run parallel tests (if pytest-xdist installed)
pytest tests/ -n auto
```

### Quick Validation

For rapid development feedback:

```bash
# Quick smoke test
python tests/validate_framework.py

# This runs basic import and functionality checks
# Much faster than full test suite
```

## 🔧 Test Configuration

### Fixtures (`conftest.py`)

The test suite uses several shared fixtures:

#### Data Fixtures
- **`classification_data`** - Multi-class classification dataset (200 samples, 10 features, 3 classes)
- **`regression_data`** - Regression dataset (200 samples, 5 features, continuous targets)
- **`time_series_data`** - Sequential data for temporal modeling
- **`text_data`** - Sample text data for NLP testing
- **`image_data`** - Synthetic image data for vision tasks

#### Environment Fixtures  
- **`temp_dir`** - Temporary directory for test outputs
- **`config_files`** - Sample configuration files
- **`mock_model_path`** - Temporary model save locations

#### Framework Fixtures
- **`sklearn_models`** - Pre-configured sklearn models for testing
- **`torch_models`** - PyTorch model configurations
- **`tf_models`** - TensorFlow model setups

### Test Markers

Tests are marked for selective execution:

```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.gpu           # Requires GPU
@pytest.mark.integration   # End-to-end tests
@pytest.mark.unit          # Unit tests
@pytest.mark.sklearn       # sklearn-specific
@pytest.mark.torch         # PyTorch-specific
@pytest.mark.tensorflow    # TensorFlow-specific
```

## 📋 Test Details

### Configuration Tests (`test_config.py`)

**Purpose**: Validate the configuration system that drives the framework

**Key Test Areas**:
- ✅ Basic configuration creation and validation
- ✅ YAML/JSON configuration file loading
- ✅ Environment variable override handling
- ✅ Configuration merging and inheritance
- ✅ Pydantic validation and error handling
- ✅ Framework-specific configuration schemas
- ✅ Experiment tracking configuration

**Example Test**:
```python
def test_config_from_yaml(self):
    """Test loading configuration from YAML file."""
    config = DLTConfig.from_yaml('config/config_dev.yaml')
    assert config.model_type is not None
    assert config.experiment['name'] is not None
```

### Model Tests (`test_model.py`)

**Purpose**: Ensure model creation, training, and management work correctly

**Key Test Areas**:
- ✅ Model factory pattern across frameworks (sklearn, PyTorch, TensorFlow)
- ✅ Model training with different optimizers and hyperparameters
- ✅ Model evaluation and metric calculation
- ✅ Model serialization and deserialization
- ✅ Multi-framework model comparison
- ✅ Model checkpointing and resuming
- ✅ Transfer learning capabilities

**Example Test**:
```python
def test_sklearn_model_training(self):
    """Test sklearn model training workflow."""
    config = DLTConfig(
        model_type='sklearn.ensemble.RandomForestClassifier',
        model_params={'n_estimators': 10, 'random_state': 42}
    )
    model = DLTModel.from_config(config)
    X, y = make_classification(n_samples=100, n_features=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
```

### Pipeline Tests (`test_pipeline.py`)

**Purpose**: Validate data processing and pipeline functionality

**Key Test Areas**:
- ✅ Data loading from various sources (CSV, Parquet, databases)
- ✅ Data preprocessing and feature engineering
- ✅ Data splitting strategies (train/val/test, cross-validation)
- ✅ Pipeline serialization and reproducibility
- ✅ Streaming data handling
- ✅ Data validation and quality checks
- ✅ Custom transformation pipeline creation

### Trainer Tests (`test_trainer.py`)

**Purpose**: Test the training orchestration and optimization systems

**Key Test Areas**:
- ✅ Training loop execution across frameworks
- ✅ Hyperparameter optimization (grid search, random search, Bayesian)
- ✅ Early stopping and learning rate scheduling
- ✅ Multi-GPU and distributed training coordination
- ✅ Training metrics logging and visualization
- ✅ Model checkpointing during training
- ✅ Resume training from checkpoints

### Utility Tests (`test_utils.py`)

**Purpose**: Test helper functions and utilities

**Key Test Areas**:
- ✅ Loss function implementations and gradients
- ✅ Performance metric calculations (accuracy, precision, recall, F1, AUC)
- ✅ Data manipulation and transformation utilities
- ✅ File I/O and serialization helpers
- ✅ Logging and monitoring utilities
- ✅ Mathematical and statistical functions

### Integration Tests (`test_integration.py`)

**Purpose**: End-to-end workflow testing

**Key Test Areas**:
- ✅ Complete training workflows from data to deployment
- ✅ Multi-step pipelines with data preprocessing and model training
- ✅ Experiment tracking and reproducibility
- ✅ Model comparison across different algorithms
- ✅ Real dataset integration testing
- ✅ API integration with external services
- ✅ Configuration-driven experiment execution

**Example Integration Test**:
```python
def test_complete_classification_workflow(self, classification_data):
    """Test complete classification workflow."""
    # Load configuration
    config = DLTConfig.from_yaml('config/examples.yaml')
    
    # Train model
    results = train(config, classification_data)
    
    # Evaluate model
    metrics = evaluate(results['model'], classification_data['test'])
    
    # Make predictions
    predictions = predict(results['model'], classification_data['test'][0])
    
    # Assertions
    assert metrics['accuracy'] > 0.7
    assert len(predictions) == len(classification_data['test'][1])
```

## 🔍 Test Coverage

The test suite aims for comprehensive coverage across:

### Framework Coverage
- **sklearn**: ✅ Complete (RandomForest, SVM, LogisticRegression, etc.)
- **PyTorch**: ✅ Neural networks, optimizers, loss functions
- **TensorFlow**: ✅ Keras models, custom training loops
- **XGBoost**: ✅ Gradient boosting models
- **LightGBM**: ✅ Efficient gradient boosting

### Functionality Coverage
- **Configuration**: 95%+ line coverage
- **Model Management**: 90%+ line coverage  
- **Data Pipeline**: 85%+ line coverage
- **Training System**: 90%+ line coverage
- **Utilities**: 95%+ line coverage
- **Integration**: 80%+ workflow coverage

### Platform Coverage
- **Operating Systems**: Linux, macOS, Windows
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **Hardware**: CPU, GPU (CUDA), Apple M1/M2

## 🚨 Continuous Integration

### GitHub Actions Workflow

Tests run automatically on:
- **Push to main branch**
- **Pull requests**
- **Scheduled daily runs**
- **Manual triggers**

### Test Matrix
```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10, 3.11]
    os: [ubuntu-latest, macos-latest, windows-latest]
    framework: [sklearn, torch, tensorflow]
```

### Quality Gates
- ✅ All tests must pass
- ✅ Coverage must be > 85%
- ✅ No critical security vulnerabilities
- ✅ Code style checks (black, flake8)
- ✅ Type checking (mypy)

## 🐛 Debugging Failed Tests

### Common Issues and Solutions

#### Import Errors
```bash
# Missing dependencies
pip install -e ".[dev]"  # Install dev dependencies
pip install -e ".[test]" # Install test dependencies
```

#### GPU-Related Failures
```bash
# Skip GPU tests if no GPU available
pytest tests/ -m "not gpu"

# Or run CPU-only mode
CUDA_VISIBLE_DEVICES="" pytest tests/
```

#### Framework-Specific Issues
```bash
# Test specific framework
pytest tests/ -m "sklearn"  # Only sklearn tests
pytest tests/ -m "not torch"  # Skip PyTorch tests
```

### Test Debugging Tips

1. **Run single test with verbose output**:
   ```bash
   pytest tests/test_model.py::TestDLTModel::test_sklearn_model_training -v -s
   ```

2. **Use pytest's built-in debugging**:
   ```bash
   pytest --pdb tests/test_config.py  # Drop into debugger on failure
   ```

3. **Check test logs**:
   ```bash
   pytest tests/ --log-cli-level=DEBUG  # Show debug logs
   ```

4. **Disable warnings for cleaner output**:
   ```bash
   pytest tests/ --disable-warnings
   ```

## 🤝 Contributing Tests

### Writing New Tests

1. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `TestSomething`
   - Test methods: `test_something_specific`

2. **Use appropriate fixtures**:
   ```python
   def test_model_training(self, classification_data):
       # Use shared fixtures from conftest.py
   ```

3. **Add proper markers**:
   ```python
   @pytest.mark.slow
   @pytest.mark.sklearn
   def test_large_dataset_training(self):
       # Long-running sklearn test
   ```

4. **Document test purpose**:
   ```python
   def test_config_validation(self):
       """Test that invalid configurations raise appropriate errors."""
   ```

### Test Quality Standards

- ✅ **Clear test names** that describe what is being tested
- ✅ **Single responsibility** - each test should test one thing
- ✅ **Proper assertions** with meaningful error messages
- ✅ **Test isolation** - tests should not depend on each other
- ✅ **Realistic data** - use fixtures that represent real use cases
- ✅ **Error testing** - test both success and failure scenarios

### Adding Test Data

1. **Small datasets**: Include in `conftest.py` fixtures
2. **Large datasets**: Download programmatically or use mock data
3. **External data**: Document requirements in test docstrings

## 📊 Test Metrics and Reporting

### Coverage Reports

Generate HTML coverage report:
```bash
pytest tests/ --cov=src/dlt --cov-report=html
# Open htmlcov/index.html in browser
```

Generate terminal coverage:
```bash
pytest tests/ --cov=src/dlt --cov-report=term-missing
```

### Performance Testing

Run performance benchmarks:
```bash
pytest tests/ -m "benchmark" --benchmark-only
```

### Test Timing

Identify slow tests:
```bash
pytest tests/ --durations=10  # Show 10 slowest tests
```

## 🚀 Best Practices

### For Developers

1. **Run tests before committing**:
   ```bash
   python tests/validate_framework.py  # Quick check
   pytest tests/ --quick               # Fast test subset
   ```

2. **Write tests for new features**:
   - Add unit tests for new functions/classes
   - Add integration tests for new workflows
   - Update existing tests when changing APIs

3. **Use appropriate test markers**:
   - Mark slow tests as `@pytest.mark.slow`
   - Mark framework-specific tests appropriately
   - Mark tests requiring special hardware

### For CI/CD

1. **Parallel execution**:
   ```bash
   pytest tests/ -n auto  # Use all available CPUs
   ```

2. **Fail fast for quick feedback**:
   ```bash
   pytest tests/ -x  # Stop at first failure
   ```

3. **Generate JUnit reports**:
   ```bash
   pytest tests/ --junitxml=results.xml
   ```

## 📝 Test Maintenance

### Regular Tasks

- **Review test coverage** monthly and add tests for uncovered code
- **Update test data** when real-world data patterns change  
- **Refactor tests** to reduce duplication and improve clarity
- **Update CI configuration** for new Python/framework versions
- **Monitor test execution times** and optimize slow tests

### Deprecation Process

When deprecating functionality:

1. Mark tests with deprecation warnings
2. Keep tests running during deprecation period
3. Remove tests only after functionality is fully removed
4. Update test documentation to reflect changes

## 🎯 Summary

The DLT test suite provides comprehensive coverage of all framework components through:

- **Unit tests** for individual component reliability
- **Integration tests** for end-to-end workflow validation  
- **Performance tests** for optimization and benchmarking
- **Compatibility tests** across frameworks and platforms

The testing infrastructure ensures that the DLT framework remains:
- **Reliable** - Components work correctly under various conditions
- **Maintainable** - Changes don't break existing functionality
- **Scalable** - Performance remains acceptable as the framework grows
- **Compatible** - Works across different environments and use cases

For questions about testing or contributing new tests, please refer to the main project documentation or open an issue on GitHub.