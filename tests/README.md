# Testing and Code Coverage Guide

This guide explains how to run all tests for the E115_SMART project with a single command, including Python tests, JavaScript tests, and DataService validation.

## Prerequisites

Make sure you have the required packages installed:

```bash
pip install pre-commit
pip install pytest pytest-cov

npm install --save-dev jest jest-environment-jsdom @babel/core @babel/preset-env @babel/plugin-transform-modules-commonjs babel-jest
```

## Project Structure

The project should have the following structure:

```
E115_SMART/
├── src/
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_advanced_semantic-chunker.py
|   └── frontend/
|           └── DataService.test.js
└── pytest.ini
```

## Running Tests

### Make the test runner executable
```
chmod +x run-all-tests.sh
```

### Run all test with one command
Just run in root:
```
./run-tests.sh
```
This script will run:

- Python tests (using pytest)
- JavaScript unit tests
- JavaScript integration tests
- DataService validation script

### Running Tests Separately (Python)

To run all python tests:

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

Add the following commands to your workflow:

```bash
python -m pytest --cov=src.datapipeline --cov-report=xml
```

## Running Tests Separately(Frontend)

### Run only JavaScript DataService tests
```
npx jest tests/frontend/DataService.test.js
```

### JavaScript Integration Tests
```
npx jest tests/frontend/DataService.integration.test.js
```

### Data Service Validation
```
node tests/frontend/validateDataService.js
```

### JavaScript Unit tests ReportsService
```
npx jest tests/frontend/ReportsService.test.js
```

###  JavaScript Integration tests ReportsService
```
npx jest tests/frontend/ReportsService.integration.test.js
```

###  JavaScript Validation ReportsService
```
node tests/frontend/validateReportsService.js
```
