import pytest
from pathlib import Path
import shutil

@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """
    Pytest fixture to provide the absolute path to the project's root directory.
    Assumes conftest.py is in the 'tests' directory, which is a sub-directory of the project root.
    """
    return Path(__file__).parent.parent.resolve()

@pytest.fixture(scope="session")
def test_config_file_path(project_root_path: Path) -> Path:
    """
    Pytest fixture to provide the path to the test configuration file.
    """
    return project_root_path / "tests" / "config_test.yaml"

@pytest.fixture(scope="session") # Używamy scope="session", aby katalog był tworzony raz na sesję
def test_artifacts_root_dir(project_root_path: Path) -> Path:
    """
    Pytest fixture to create and provide the path to the root directory
    for storing test artifacts (logs, checkpoints, etc.).
    This directory is 'tests/test_artifacts/' relative to the project root.
    It will be created if it doesn't exist.
    """
    artifacts_dir = project_root_path / "tests" / "test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test artifacts will be stored in: {artifacts_dir}")
    return artifacts_dir

@pytest.fixture(scope="function") # Scope "function" dla czystego katalogu na każdy test
def unique_test_run_artifacts_dir(test_artifacts_root_dir: Path, request) -> Path:
    """
    Pytest fixture to create a unique sub-directory within the test_artifacts_root_dir
    for each test function that uses it. This helps isolate artifacts from different tests.
    The directory is named after the test function.
    The directory is cleaned up before the test if it already exists.
    """
    # request.node.name daje nazwę aktualnie uruchomionego testu
    test_name = request.node.name.replace("[", "_").replace("]", "").replace("/", "_") # Sanitize name
    run_specific_dir = test_artifacts_root_dir / test_name

    if run_specific_dir.exists():
        print(f"Cleaning up existing artifact directory: {run_specific_dir}")
        shutil.rmtree(run_specific_dir)
    run_specific_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created unique artifact directory for test '{request.node.name}': {run_specific_dir}")

    return run_specific_dir
