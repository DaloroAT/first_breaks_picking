import os
import shutil
from pathlib import Path

import pytest

from first_breaks.const import PROJECT_ROOT


def find_code_block(file: Path, start_indicator: str, end_indicator: str) -> str:
    with open(file) as f:
        text = f.readlines()

    i = text.index(start_indicator) + 2
    j = text.index(end_indicator) - 1

    code_block = "".join(text[i:j])

    return code_block


@pytest.mark.parametrize(
    "block_name",
    [
        "e2e-example",
        # "downloading-extra",
        "init-from-path",
        "init-from-bytes",
        "init-from-np",
        "init-from-np",
        "sgy-content",
        "create-task",
        "create-picker",
        "pick-fb",
        "picks",
        "plot-sgy",
        "plot-np",
        "plot-sgy-custom-picks",
        "plot-sgy-real-picks"
    ],
)
def test_code_blocks_in_readme(block_name: str,
                               demo_sgy: Path,
                               logs_dir_for_tests: Path) -> None:
    assert logs_dir_for_tests.exists()
    os.chdir(str(logs_dir_for_tests))
    shutil.copyfile(str(demo_sgy), str(logs_dir_for_tests / 'data.sgy'))

    start_indicator = f"[code-block-start]:{block_name}\n"
    end_indicator = f"[code-block-end]:{block_name}\n"
    code = find_code_block(PROJECT_ROOT / "README.md", start_indicator, end_indicator)
    assert code

    tmp_fname = "tmp.py"

    with open(tmp_fname, "w") as f:
        f.write(code)

    try:
        code = os.system(f"python {tmp_fname}")
        assert code == 0
    finally:
        Path(tmp_fname).unlink()
