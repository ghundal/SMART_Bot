# Python and TypeScript Linting and Formatting Guide

This guide covers how to set up and use linters and formatters to maintain code quality in this project.

## Setup

Run the setup script to install all linters and formatters:

```bash
pip install pre-commit
chmod +x linter/setup.sh
./linter/setup.sh
```

## Tools Included

### Python

- **Black**: Code formatter (line length: 100)
- **isort**: Import statement formatter
- **Flake8**: Style guide enforcer
- **MyPy**: Type checker

### TypeScript/JavaScript

- **ESLint**: Static code analyzer (using flat config format)
- **Prettier**: Code formatter

### Docker and YAML

- **hadolint**: Dockerfile linter
- **yamllint**: YAML linter

## Running All Linters and Formatters

You can run all linters and formatters with a single command:

```bash
./linter/run-all-linters.sh
```

## Running Individual Tools

### Python (with Pipenv)

```bash
./linter/lint-and-format.sh
```

### TypeScript/JavaScript (with npm)

```bash
./linter/ts-lint-and-format.sh
```

### Docker Files

```bash
./linter/docker-lint.sh
```

## Configuration Files

All configuration files are now at the root level of the project:

- Python:

  - `pyproject.toml` (Black, isort)
  - `.flake8`
  - `mypy.ini`

- TypeScript:

  - `eslint.config.js` (ESLint flat config)
  - `.prettierrc`

- Docker/YAML:
  - `.yamllint`

## Flat Config Format for ESLint

This project uses ESLint's new flat config format, which is the default in ESLint v9+. The key differences from the traditional format are:

- Configuration is an array of config objects
- Parser settings are specified under `languageOptions`
- Plugins are specified as objects instead of strings
- No `extends` property - configs are composed by adding objects to the array
- No `root` property (flat configs are always treated as root)

For more details, see the [ESLint Flat Config Migration Guide](https://eslint.org/docs/latest/use/configure/migration-guide).

## Commit

Run pre-commit locally before pushing:

```
pre-commit run --all-files
```
