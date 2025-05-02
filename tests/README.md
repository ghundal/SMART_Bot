# Testing and Code Coverage Guide

This guide explains how to run the test suite for the data pipeline project and generate code coverage reports.

## Prerequisites

Make sure you have the required packages installed:

```bash
pip install pytest pytest-cov
```

## Project Structure

The project should have the following structure:

```
E115_SMART/
├── src/
│   ├── datapipeline/
│   │   ├── __init__.py
│   │   ├── datapipeline.py
│   │   └── Advanced_semantic_chunker.py
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_advanced_semantic-chunker.py
└── pytest.ini
```

## Running Tests

### Basic Test Execution

To run all tests:

```bash
python -m pytest
```

To run tests with detailed output:

```bash
python -m pytest -v
```

To run a specific test file or to output in a text file:

```bash
python -m pytest tests/unit/test_data_pipeline.py
python -m pytest tests/unit/test_data_pipeline.py > output.txt
```

To run a specific test class or function:

```bash
python -m pytest tests/unit/test_data_pipeline.py::TestDataPipeline
python -m pytest tests/unit/test_data_pipeline.py::TestDataPipeline::test_clean_chunks
```

### Logging

The tests output logs at the INFO level. The configuration is in the `pytest.ini` file:

```ini
log_cli = True
log_cli_level = INFO
```

## Code Coverage

### Basic Coverage Report

To run tests with coverage reporting:

```bash
python -m pytest --cov=src.datapipeline
```

### HTML Coverage Report

To generate a detailed HTML coverage report:

```bash
python -m pytest --cov=src.datapipeline --cov-report=html
```

This creates an `htmlcov` directory. Open `htmlcov/index.html` in a web browser to view the report.

### XML Coverage Report (for CI/CD tools)

To generate an XML coverage report for integration with CI/CD tools:

```bash
python -m pytest --cov=src.datapipeline --cov-report=xml
```

### Excluding Code from Coverage

You can exclude specific lines or functions from coverage by adding `# pragma: no cover` comments in your code.

## Continuous Integration

If you're using CI/CD pipelines, add the following commands to your workflow:

```bash
python -m pytest --cov=src.datapipeline --cov-report=xml
```

This will generate a coverage.xml file that can be used by services like Codecov or SonarQube.

## Troubleshooting

### Module Not Found Errors

If you encounter module not found errors, make sure your Python path includes the project root:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Coverage Not Working

If coverage isn't working correctly, check that your package structure is correct and that you're using the right import paths. The `--cov` parameter should match your actual module structure.
