#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS_DIR="${REQUIREMENTS_DIR:-"$ROOT_DIR/requirements"}"
VENVS_DIR="${VENVS_DIR:-"$ROOT_DIR/.venvs"}"
INSTALL_NEW=1
SYNC_EXISTING=0

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_venvs.sh [options] [env-or-requirements ...]

Creates one virtualenv per requirements/requirements-*.txt file.
Virtualenvs are written to .venvs/<name>, where <name> comes from the
requirements filename.

Options:
  --no-install       Create missing virtualenvs without installing packages.
  --sync-existing    Run pip install -r even when the virtualenv already exists.
  -h, --help         Show this help.

Examples:
  scripts/setup_venvs.sh
  scripts/setup_venvs.sh darts chronos
  scripts/setup_venvs.sh --sync-existing requirements/requirements-darts.txt
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

python_from_requirement() {
  local requirement_file="$1"
  local requested_version major_minor candidate

  requested_version="$(awk -F: '/^#python:/ { gsub(/[[:space:]]/, "", $2); print $2; exit }' "$requirement_file")"

  if [[ -n "$requested_version" ]]; then
    major_minor="$(printf '%s\n' "$requested_version" | awk -F. '{ print $1 "." $2 }')"
    for candidate in "python${requested_version}" "python${major_minor}" "python3"; do
      if command -v "$candidate" >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf 'python3\n'
    return 0
  fi

  die "python3 not found for $requirement_file"
}

requirement_for_arg() {
  local arg="$1"

  if [[ -f "$arg" ]]; then
    printf '%s\n' "$arg"
    return 0
  fi

  if [[ -f "$REQUIREMENTS_DIR/$arg" ]]; then
    printf '%s\n' "$REQUIREMENTS_DIR/$arg"
    return 0
  fi

  if [[ -f "$REQUIREMENTS_DIR/requirements-$arg.txt" ]]; then
    printf '%s\n' "$REQUIREMENTS_DIR/requirements-$arg.txt"
    return 0
  fi

  die "requirements file not found for '$arg'"
}

venv_name_from_requirement() {
  local requirement_file="$1"
  local filename

  filename="$(basename "$requirement_file")"
  filename="${filename#requirements-}"
  filename="${filename%.txt}"
  printf '%s\n' "$filename"
}

setup_venv() {
  local requirement_file="$1"
  local name venv_path python_bin created

  name="$(venv_name_from_requirement "$requirement_file")"
  venv_path="$VENVS_DIR/$name"
  python_bin="$(python_from_requirement "$requirement_file")"
  created=0

  if [[ -d "$venv_path" ]]; then
    printf '[skip] %s already exists at %s\n' "$name" "$venv_path"
  else
    printf '[create] %s using %s\n' "$venv_path" "$python_bin"
    "$python_bin" -m venv "$venv_path"
    created=1
  fi

  if [[ "$INSTALL_NEW" -eq 1 && ( "$created" -eq 1 || "$SYNC_EXISTING" -eq 1 ) ]]; then
    printf '[install] %s <- %s\n' "$name" "$requirement_file"
    "$venv_path/bin/python" -m pip install --upgrade pip
    "$venv_path/bin/python" -m pip install -r "$requirement_file"
  fi
}

main() {
  local args=()
  local requirement_files=()
  local arg file

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --no-install)
        INSTALL_NEW=0
        ;;
      --sync-existing)
        SYNC_EXISTING=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        args+=("$1")
        ;;
    esac
    shift
  done

  [[ -d "$REQUIREMENTS_DIR" ]] || die "requirements directory not found: $REQUIREMENTS_DIR"
  mkdir -p "$VENVS_DIR"

  if [[ "${#args[@]}" -eq 0 ]]; then
    while IFS= read -r file; do
      requirement_files+=("$file")
    done < <(find "$REQUIREMENTS_DIR" -maxdepth 1 -type f -name 'requirements-*.txt' | sort)
  else
    for arg in "${args[@]}"; do
      requirement_files+=("$(requirement_for_arg "$arg")")
    done
  fi

  [[ "${#requirement_files[@]}" -gt 0 ]] || die "no requirements files found in $REQUIREMENTS_DIR"

  for file in "${requirement_files[@]}"; do
    setup_venv "$file"
  done
}

main "$@"
