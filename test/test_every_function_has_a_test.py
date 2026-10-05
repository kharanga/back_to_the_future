import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
PACKAGES = ("listings", "tag_analyzer")

IMPORTS_UNSLOTH = "the module imports unsloth, which does not install on the laptop"
IMPORTS_UNSLOTH_MODULES = "the module imports listings.model, which imports unsloth"
RUNS_THE_MODEL = "the module drives the fine-tuned model and is GPU-box only; the validator never imports it"

FUNCTIONS_THAT_CANNOT_BE_TESTED_ON_THE_LAPTOP = {
    "listings/model.py::limit_image_pixels": IMPORTS_UNSLOTH,
    "listings/model.py::load_base_model": IMPORTS_UNSLOTH,
    "listings/model.py::add_lora_to_language_layers": IMPORTS_UNSLOTH,
    "listings/model.py::load_finetuned_model": IMPORTS_UNSLOTH,
    "listings/train.py::with_loaded_photos": IMPORTS_UNSLOTH,
    "listings/train.py::trainable_conversation": IMPORTS_UNSLOTH,
    "listings/train.py::PhotosLoadedOnAccess.__init__": IMPORTS_UNSLOTH,
    "listings/train.py::PhotosLoadedOnAccess.__len__": IMPORTS_UNSLOTH,
    "listings/train.py::PhotosLoadedOnAccess.__getitem__": IMPORTS_UNSLOTH,
    "listings/train.py::training_config": IMPORTS_UNSLOTH,
    "listings/train.py::main": IMPORTS_UNSLOTH,
    "listings/eval.py::evaluate_example": IMPORTS_UNSLOTH_MODULES,
    "listings/eval.py::main": IMPORTS_UNSLOTH_MODULES,
    "listings/generate.py::generation_messages": RUNS_THE_MODEL,
    "listings/generate.py::generate_first_line": RUNS_THE_MODEL,
}


def source_modules() -> list[Path]:
    return [module for package in PACKAGES for module in sorted((REPO_ROOT / package).glob("*.py"))]


def function_definitions(nodes: list[ast.stmt]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def qualified_function_names(module: Path) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names_with_lines = [(function.lineno, function.name) for function in function_definitions(tree.body)]
    for class_definition in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        names_with_lines += [
            (method.lineno, f"{class_definition.name}.{method.name}")
            for method in function_definitions(class_definition.body)
        ]
    return [name for _, name in sorted(names_with_lines)]


def snake_case(class_name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()


def function_and_class_names(module: Path) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    class_names_with_lines = [
        (node.lineno, snake_case(node.name)) for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    function_names_with_lines = [(function.lineno, function.name) for function in function_definitions(tree.body)]
    return [name for _, name in sorted(class_names_with_lines + function_names_with_lines)]


def function_id(module: Path, function_name: str) -> str:
    return f"{module.relative_to(REPO_ROOT).as_posix()}::{function_name}"


def every_function_id() -> list[str]:
    return [function_id(module, name) for module in source_modules() for name in qualified_function_names(module)]


def file_of_tests_for(module: Path) -> Path:
    return TEST_DIR / module.parent.name / f"test_{module.stem}.py"


def names_of_tests_in(test_file: Path) -> list[str]:
    if not test_file.exists():
        return []
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    return [function.name for function in function_definitions(tree.body) if function.name.startswith("test_")]


def prefix_of_tests_for(function_name: str) -> str:
    return f"test_{function_name.replace('.', '_')}_"


def function_tested_by(test_name: str, function_names: list[str]) -> str | None:
    named_after = [name for name in function_names if test_name.startswith(prefix_of_tests_for(name))]
    return max(named_after, key=len, default=None)


def functions_that_need_a_test() -> list[tuple[Path, str]]:
    return [
        (module, name)
        for module in source_modules()
        for name in qualified_function_names(module)
        if function_id(module, name) not in FUNCTIONS_THAT_CANNOT_BE_TESTED_ON_THE_LAPTOP
    ]


def modules_with_a_test_file() -> list[Path]:
    return [module for module in source_modules() if file_of_tests_for(module).exists()]


def readable_id(value) -> str:
    return value.relative_to(REPO_ROOT).as_posix() if isinstance(value, Path) else value


@pytest.mark.parametrize(("module", "function_name"), functions_that_need_a_test(), ids=readable_id)
def test_every_function_has_a_test_named_after_it(module, function_name):
    prefix = prefix_of_tests_for(function_name)
    tests_for_function = [name for name in names_of_tests_in(file_of_tests_for(module)) if name.startswith(prefix)]
    assert tests_for_function, f"no test named {prefix}<what_it_does> in {readable_id(file_of_tests_for(module))}"


@pytest.mark.parametrize("module", modules_with_a_test_file(), ids=readable_id)
def test_every_test_is_named_after_a_function_or_class_in_its_module(module):
    names = function_and_class_names(module)
    unnamed = [name for name in names_of_tests_in(file_of_tests_for(module)) if function_tested_by(name, names) is None]
    assert unnamed == []


@pytest.mark.parametrize("module", modules_with_a_test_file(), ids=readable_id)
def test_tests_follow_the_order_of_the_functions_and_classes_in_the_module(module):
    names = function_and_class_names(module)
    tested_in_order = [function_tested_by(name, names) for name in names_of_tests_in(file_of_tests_for(module))]
    order_in_test_file = list(dict.fromkeys(name for name in tested_in_order if name is not None))
    order_in_module = [name for name in names if name in order_in_test_file]
    assert order_in_test_file == order_in_module


def test_every_exempt_function_still_exists_in_the_source():
    stale_exemptions = sorted(set(FUNCTIONS_THAT_CANNOT_BE_TESTED_ON_THE_LAPTOP) - set(every_function_id()))
    assert stale_exemptions == []


def test_no_init_file_under_test_shadows_the_real_packages():
    assert list(TEST_DIR.rglob("__init__.py")) == []
