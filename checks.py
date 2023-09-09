from pathlib import Path

import tomli


def compare_main_and_gpu_tomls() -> None:
    with open(Path(__file__).parent / "pyproject.toml", "rb") as fin:
        pyproject_main = tomli.load(fin)

    with open(Path(__file__).parent / "pyproject_gpu.toml", "rb") as fin:
        pyproject_gpu = tomli.load(fin)

    deps_main = set(pyproject_main["project"].pop("dependencies"))
    deps_gpu = set(pyproject_gpu["project"].pop("dependencies"))

    # Analyse onnx version

    onnx_main = [dep for dep in deps_main if dep.startswith("onnxruntime")]
    onnx_gpu = [dep for dep in deps_gpu if dep.startswith("onnxruntime-gpu")]

    if not onnx_main:
        raise EnvironmentError("'onnxruntime' is not in dependencies in 'pyproject.toml'")

    if not onnx_gpu:
        raise EnvironmentError("'onnxruntime-gpu' is not in dependencies in 'pyproject_gpu.toml'")

    onnx_main = onnx_main[0]
    onnx_gpu = onnx_gpu[0]

    onnx_main = onnx_main.split("==")
    onnx_gpu = onnx_gpu.split("==")

    if not onnx_main or not onnx_gpu:
        raise EnvironmentError("Please, fix versions for 'onnxruntime' and 'onnxruntime-gpu'")

    if onnx_main[1] != onnx_gpu[1]:
        raise EnvironmentError("Versions of 'onnxruntime' and 'onnxruntime-gpu' are different")

    # Compare other deps

    difference = deps_main.symmetric_difference(deps_gpu)

    if not (all("onnxruntime" in dep for dep in difference) and (deps_main - difference) == (deps_gpu - difference)):
        raise EnvironmentError("Only difference between 'onnxruntime' is available")

    # Compare project names

    project_name_main = pyproject_main["project"].pop("name")
    project_name_gpu = pyproject_gpu["project"].pop("name")

    if project_name_main + "-gpu" != project_name_gpu:
        raise EnvironmentError(f"Wrong project names: {project_name_main} and {project_name_gpu}")

    # Compare app names

    app_name_main = pyproject_main["tool"]["briefcase"].pop("project_name")
    app_name_gpu = pyproject_gpu["tool"]["briefcase"].pop("project_name")

    if app_name_main + "GPU" != app_name_gpu:
        raise EnvironmentError(f"Wrong app names: {app_name_main} and {app_name_gpu}")

    app_formal_name_main = pyproject_main["tool"]["briefcase"]["app"]["first_breaks"].pop("formal_name")
    app_formal_name_gpu = pyproject_gpu["tool"]["briefcase"]["app"]["first_breaks"].pop("formal_name")

    if app_formal_name_main + "GPU" != app_formal_name_gpu:
        raise EnvironmentError(f"Wrong app names: {app_formal_name_main} and {app_formal_name_gpu}")

    # Compare rest of the tomls

    del pyproject_main["project"]["description"]
    del pyproject_gpu["project"]["description"]

    if pyproject_main != pyproject_gpu:
        raise EnvironmentError('Main config and gpu config are different')

    print("TOMLs are correct")


def compare_versions_of_repo() -> None:
    with open(Path(__file__).parent / "first_breaks/VERSION", 'r') as fin:
        project_version = fin.read()

    with open(Path(__file__).parent / "pyproject.toml", 'rb') as fin:
        briefcase_version = tomli.load(fin)["tool"]["briefcase"]["version"]

    if project_version != briefcase_version:
        raise EnvironmentError(f"Version in 'briefcase' config and version of project are different. "
                               f"Briefcase: {briefcase_version}, project: {project_version}")

    print("Versions are the same")


if __name__ == "__main__":
    compare_main_and_gpu_tomls()
    compare_versions_of_repo()
