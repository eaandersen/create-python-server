import json
import re
import subprocess
import sys
from pathlib import Path
import os

import click
import toml
from packaging.version import parse
from faker import Faker

MIN_UV_VERSION = "0.4.10"


class PyProject:
    def __init__(self, path: Path):
        self.data = toml.load(path)

    @property
    def name(self) -> str:
        return self.data["project"]["name"]

    @property
    def first_binary(self) -> str | None:
        scripts = self.data["project"].get("scripts", {})
        return next(iter(scripts.keys()), None)


def check_uv_version(required_version: str) -> str | None:
    """Check if uv is installed and has minimum version"""
    try:
        result = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        match = re.match(r"uv (\d+\.\d+\.\d+)", version)
        if match:
            version_num = match.group(1)
            if parse(version_num) >= parse(required_version):
                return version
        return None
    except subprocess.CalledProcessError:
        click.echo("❌ Error: Failed to check uv version.", err=True)
        sys.exit(1)
    except FileNotFoundError:
        return None


def ensure_uv_installed() -> None:
    """Ensure uv is installed at minimum version"""
    if check_uv_version(MIN_UV_VERSION) is None:
        click.echo(
            f"❌ Error: uv >= {MIN_UV_VERSION} is required but not installed.", err=True
        )
        click.echo("To install, visit: https://github.com/astral-sh/uv", err=True)
        sys.exit(1)


def get_claude_config_path() -> Path | None:
    """Get the Claude config directory based on platform"""
    if sys.platform == "win32":
        path = Path(Path.home(), "AppData", "Roaming", "Claude")
    elif sys.platform == "darwin":
        path = Path(Path.home(), "Library", "Application Support", "Claude")
    else:
        return None

    if path.exists():
        return path
    return None


def has_claude_app() -> bool:
    return get_claude_config_path() is not None


def update_claude_config(project_name: str, project_path: Path) -> bool:
    """Add the project to the Claude config if possible"""
    config_dir = get_claude_config_path()
    if not config_dir:
        return False

    config_file = config_dir / "claude_desktop_config.json"
    if not config_file.exists():
        return False

    try:
        config = json.loads(config_file.read_text())
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        if project_name in config["mcpServers"]:
            click.echo(
                f"⚠️ Warning: {project_name} already exists in Claude.app configuration",
                err=True,
            )
            click.echo(f"Settings file location: {config_file}", err=True)
            return False

        config["mcpServers"][project_name] = {
            "command": "uv",
            "args": ["--directory", str(project_path), "run", project_name],
        }

        config_file.write_text(json.dumps(config, indent=2))
        click.echo(f"✅ Added {project_name} to Claude.app configuration")
        click.echo(f"Settings file location: {config_file}")
        return True
    except Exception:
        click.echo("❌ Failed to update Claude.app configuration", err=True)
        click.echo(f"Settings file location: {config_file}", err=True)
        return False


def get_package_directory(path: Path) -> Path:
    """Find the package directory under src/"""
    src_dir = next((path / "src").glob("*/__init__.py"), None)
    if src_dir is None:
        click.echo("❌ Error: Could not find __init__.py in src directory", err=True)
        sys.exit(1)
    return src_dir.parent


def copy_template(
    path: Path, name: str, description: str, version: str = "0.1.0"
) -> None:
    """Copy template files into src/<project_name>"""
    template_dir = Path(__file__).parent / "template"

    target_dir = get_package_directory(path)

    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Create tests directory if it doesn't exist
    tests_dir = path / "tests"
    tests_dir.mkdir(exist_ok=True)

    template_vars = {
        "binary_name": name.replace("-", "_"),  # Use name instead of reading pyproject
        "server_name": name,
        "server_version": version,
        "server_description": description,
        "server_directory": str(path.resolve()),
    }

    files = [
        ("__init__.py.jinja2", "__init__.py", target_dir),
        ("server.py.jinja2", "server.py", target_dir),
        ("README.md.jinja2", "README.md", path),
        ("pyproject.toml.jinja2", "pyproject.toml", path),
        ("tests/test_server.py.jinja2", "tests/test_server.py", path),
    ]

    try:
        for template_file, output_file, output_dir in files:
            template = env.get_template(template_file)
            rendered = template.render(**template_vars)

            out_path = output_dir / output_file
            # Ensure parent directories exist
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered)

    except Exception as e:
        click.echo(f"❌ Error: Failed to template and write files: {e}", err=True)
        sys.exit(1)


def create_project(
    path: Path,
    name: str,
    description: str,
    version: str = "0.1.0",
    claudeapp: bool = True,
) -> None:
    """Create a new project at the given path"""
    try:
        # Create project directory first
        path.mkdir(parents=True, exist_ok=True)
        click.echo(f"Created project directory at {path}")

        # Create src directory and package directory
        src_dir = path / "src"
        package_dir = src_dir / name.replace("-", "_")
        src_dir.mkdir(exist_ok=True, parents=True)
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").touch()
        click.echo(f"Created package directory at {package_dir}")

        # Copy template files
        click.echo("Copying template files...")
        copy_template(path, name, description, version)
        click.echo("Template files copied")

        # Create venv in .venv directory
        venv_path = path / ".venv"
        click.echo(f"Creating venv at {venv_path}")
        subprocess.run(
            [
                "uv",
                "venv",
                str(venv_path),
            ],
            check=True,
        )
        click.echo("Venv created")

        # Activate venv by modifying PATH and VIRTUAL_ENV
        if os.name == "nt":  # Windows
            os.environ["PATH"] = f"{venv_path}\\Scripts;{os.environ['PATH']}"
        else:  # Unix
            os.environ["PATH"] = f"{venv_path}/bin:{os.environ['PATH']}"
        os.environ["VIRTUAL_ENV"] = str(venv_path)

        # Install dependencies in activated venv
        click.echo("Installing dependencies...")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                ".",
                "mcp @ git+https://github.com/modelcontextprotocol/python-sdk.git@v1.2.0rc1",
                "pytest>=7.4.0",
                "pytest-asyncio>=0.21.1",
            ],
            cwd=path,
            check=True,
        )

        click.echo(f"\n✨ Created project {name} at {path}")

    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error: Command failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def check_package_name(name: str) -> bool:
    """Check if the package name is valid according to pyproject.toml spec"""
    if not name:
        click.echo("❌ Project name cannot be empty", err=True)
        return False
    if " " in name:
        click.echo("❌ Project name must not contain spaces", err=True)
        return False
    if not all(c.isascii() and (c.isalnum() or c in "_-.") for c in name):
        click.echo(
            "❌ Project name must consist of ASCII letters, digits, underscores, hyphens, and periods",
            err=True,
        )
        return False
    if name.startswith(("_", "-", ".")) or name.endswith(("_", "-", ".")):
        click.echo(
            "❌ Project name must not start or end with an underscore, hyphen, or period",
            err=True,
        )
        return False
    return True


def find_workspace_root() -> Path:
    """
    Find the workspace root directory.

    Looks for the main git repository root, ignoring submodules.

    Returns:
        Path: The workspace root directory
    Raises:
        ValueError: If not in a git repository
    """
    current = Path.cwd().resolve()

    # First check if we're already at root
    if (current / ".git").is_dir():
        return current

    # If not, walk up the directory tree
    while current != current.parent:
        git_path = current / ".git"
        if git_path.is_dir():
            return current
        current = current.parent

    raise ValueError(
        "Not in a git repository. Please run from within a version-controlled workspace."
    )


def generate_server_name():
    """Generate a creative server name."""
    fake = Faker()

    # List of interesting prefixes using available providers
    prefixes = [
        fake.color_name().lower(),  # e.g., 'blue', 'crimson'
        fake.word().lower(),  # e.g., 'dynamic', 'quantum'
        fake.company().split()[0].lower(),  # e.g., 'micro', 'global'
    ]

    # List of interesting suffixes
    suffixes = [
        "hub",
        "node",
        "core",
        "edge",
        "gate",
        "link",
        "port",
        "base",
    ]

    # Pick random elements
    prefix = fake.random_element(prefixes)
    suffix = fake.random_element(suffixes)

    # Clean up and combine
    name = f"{prefix}-{suffix}".replace(" ", "-")
    return name


def validate_name(ctx, param, value):
    """Validate and normalize the server name."""
    # Generate name if none provided
    if not value:
        value = generate_server_name()
        click.echo(f"Generated server name: {value}")

    # Strip existing prefix if present
    if value.startswith("server-"):
        value = value[7:]

    # Validate the base name
    if not re.match(r"^[a-zA-Z][-a-zA-Z0-9_]+$", value):
        raise click.BadParameter(
            "Server name must start with a letter and contain only letters, numbers, hyphens, and underscores"
        )

    # Return with prefix
    return f"server-{value}"


@click.command()
@click.option(
    "--path",
    type=click.Path(path_type=Path),
    help="Directory to create project in",
)
@click.option(
    "--name",
    type=str,
    help="Project name (will be prefixed with 'server-'). Leave empty for auto-generation.",
    callback=validate_name,
)
@click.option(
    "--version",
    type=str,
    help="Server version",
)
@click.option(
    "--description",
    type=str,
    help="Project description",
)
@click.option(
    "--claudeapp/--no-claudeapp",
    default=True,
    help="Enable/disable Claude.app integration",
)
def main(
    path: Path | None,
    name: str | None,
    version: str | None,
    description: str | None,
    claudeapp: bool,
) -> int:
    """Create a new MCP server project"""
    ensure_uv_installed()

    click.echo("Creating a new MCP server project using uv.")
    click.echo("This will set up a Python project with MCP dependency.")
    click.echo("\nLet's begin!\n")

    name = click.prompt("Project name (required)", type=str) if name is None else name

    if name is None:
        click.echo("❌ Error: Project name cannot be empty", err=True)
        return 1

    if not check_package_name(name):
        return 1

    description = (
        click.prompt("Project description", type=str, default="A MCP server project")
        if description is None
        else description
    )

    assert isinstance(description, str)

    # Validate version if not supplied on command line
    if version is None:
        version = click.prompt("Project version", default="0.1.0", type=str)
        assert isinstance(version, str)
        try:
            parse(version)  # Validate semver format
        except Exception:
            click.echo(
                "❌ Error: Version must be a valid semantic version (e.g. 1.0.0)",
                err=True,
            )
            return 1

    try:
        # Find workspace root and set default path
        workspace_root = find_workspace_root()

        # Default to 'servers' directory if path not specified
        if path is None:
            servers_dir = workspace_root / "servers"
            servers_dir.mkdir(exist_ok=True)
            project_path = servers_dir.resolve() / name  # Use resolved path
        else:
            project_path = path

        # Ask the user if the path is correct if not specified on command line
        if path is None:
            click.echo(f"Project will be created at: {project_path}")
            if not click.confirm("Is this correct?", default=True):
                project_path = Path(
                    click.prompt(
                        "Enter the correct path", type=click.Path(path_type=Path)
                    )
                )

        if project_path is None:
            click.echo("❌ Error: Invalid path. Project creation aborted.", err=True)
            return 1

        project_path = project_path.resolve()  # Ensure absolute path

        # Change to workspace root before creating project
        os.chdir(workspace_root)  # Add this line

        create_project(project_path, name, description, version, claudeapp)

        return 0

    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        return 1
