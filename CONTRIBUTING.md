# CONTRIBUTING 

## Requires

- Python 3.12+
-  [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Dev setup

Clone the repository

```sh
# Install all dependencies
uv run sync --dev
```

## Testing

```sh
uv run pytest
```

## Linting, formatting, and type checking

```sh
uv run ruff format
```

```sh
uv run ruff check --fix
```

```sh
uv run mypy src/
```

## Branching

- Branch off `dev`

## Pull requests



## CI/CD


