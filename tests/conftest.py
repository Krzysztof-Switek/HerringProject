import pytest
from pathlib import Path
import shutil
import sys

@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """
    Pytest fixture to provide the absolute path to the project's root directory.
    Assumes conftest.py is in the 'tests' directory, which is a sub-directory of the project root.
    """
    return Path(__file__).parent.parent.resolve()

@pytest.fixture(scope="session", autouse=True)
def add_src_to_sys_path(project_root_path):
    """
    Fixture to add the 'src' directory to sys.path for the test session.
    This helps in resolving imports from the 'src' directory.
    'autouse=True' ensures this fixture is used automatically for all tests in the session.
    """
    src_path = project_root_path / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    # Można też dodać sam project_root, jeśli importy są typu `from src.engine...`
    # Jeśli importy są `from engine...` (po dodaniu src do path), to powyższe jest OK.


@pytest.fixture(scope="session")
def test_config_file_path(project_root_path: Path) -> Path:
    """
    Pytest fixture to provide the path to the test configuration file.
    """
    path = project_root_path / "tests" / "config_test.yaml"
    # Upewnijmy się, że plik istnieje, zanim testy zaczną go używać
    # Chociaż jeśli jest tworzony w poprzednim kroku, to powinien istnieć.
    # assert path.exists(), f"Test config file not found at: {path}" # Można dodać, ale nie jest to rola fixture
    return path

@pytest.fixture(scope="session")
def test_artifacts_root_dir(project_root_path: Path) -> Path:
    """
    Pytest fixture to create and provide the path to the root directory
    for storing test artifacts (logs, checkpoints, etc.).
    This directory is 'project_root/test_artifacts/'.
    It will be created if it doesn't exist.
    """
    # Zmieniono ścieżkę, aby była w głównym katalogu projektu, a nie w tests/
    # To ułatwia inspekcję i jest bardziej standardowe.
    artifacts_dir = project_root_path / "test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test artifacts root directory: {artifacts_dir}")
    return artifacts_dir

@pytest.fixture(scope="function")
def unique_test_run_artifacts_dir(test_artifacts_root_dir: Path, request) -> Path:
    """
    Pytest fixture to create a unique sub-directory within the test_artifacts_root_dir
    for each test function that uses it. This helps isolate artifacts from different tests.
    The directory is named after the test function (sanitized).
    The directory is cleaned up (removed and recreated) before the test run.
    """
    test_name = request.node.name
    # Sanitize the test name to be a valid directory name
    # Replace characters that are problematic in directory names
    sanitized_test_name = "".join(c if c.isalnum() else "_" for c in test_name)

    run_specific_dir = test_artifacts_root_dir / sanitized_test_name

    if run_specific_dir.exists():
        print(f"Cleaning up existing artifact directory: {run_specific_dir}")
        shutil.rmtree(run_specific_dir)
    run_specific_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created unique artifact directory for test '{test_name}': {run_specific_dir}")

    return run_specific_dir
