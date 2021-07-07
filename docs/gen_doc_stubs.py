#!/usr/bin/env python
"""
This script is used to generated the code reference pages for mkdocstrings.
"""

from pathlib import Path
import mkdocs_gen_files


# only allow certain directory trees from vast_pipeline.
exclude_dirs = ['data']

problem_files = ['__init__.py']

for path in Path("vasttools").glob("**/*.py"):

    if len(path.parts) > 2 and path.parts[1] in exclude_dirs:
        continue

    if path.name in problem_files:
        continue

    doc_path = Path(
        "reference", path.relative_to("vasttools")
    ).with_suffix(".md")

    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = ".".join(path.relative_to(".").with_suffix("").parts)
        print("::: " + ident, file=f)

    mkdocs_gen_files.set_edit_path(doc_path, Path("..", path))
