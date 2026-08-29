# CONTRIBUTING 

## Dev setup

install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
# Clone the repo and go to the directory
git clone https://github.com/Sci-Fi-Parser/Sci-Fi-Parser.git
cd Sci-fi-parser

# Install all dependencies
uv run sync --dev
```

## Testing

```sh
uv run pytest
```

## Linting, formatting, and type checking

```sh
# Ruff config is defined in ruff.toml
uv run ruff check --fix

uv run ruff format

uv run mypy src/
```

## Branching

- Feature branch off `dev`
- Merge feature branches to `dev`
- For release, merge `dev` to `main`

## CI/CD

- CI runs on pull requests to main and dev
- Runs linter and tests. Type checking is not done.

- CD runs on tag push `vX.Y.Z`
    - The tag is first pushed to TestPyPI and then PyPI