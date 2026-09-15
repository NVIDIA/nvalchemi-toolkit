# NVIDIA ALCHEMI Toolkit Contributor's Guide

> [!IMPORTANT]
> During the initial public beta, `nvalchemi-toolkit` will not be accepting
> direct code contributions. This message will be removed once we are ready
> to review and accept pull requests from the public.

## Documentation

First and foremost, familiarize yourself with the existing documentation, both
in the `sphinx` docs, as well in kernel docstrings. In particular, see the kernel
style guide to ensure consistent variable names and conventions across the codebase.

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the
contribution is your original work, or you have rights to submit it under the same
license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license
    document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to
    submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
    is covered under an appropriate open source license and I have the right under that
    license to submit that work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am permitted to submit under a
    different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified
    (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I submit with
    it, including my sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s) involved.

  ```

### Pre-commit

For ALCHEMI Toolkit development, [pre-commit](https://pre-commit.com/) is **required**.
This will not only help developers pass the CI pipeline, but also accelerate reviews.
Contributions that have not used pre-commit will *not be reviewed*.

`pre-commit` is installed as part of the `dev` optional dependencies defined in `pyproject.toml`.
If using `uv`, then running `uv sync --extra cuXX --group docs` (or `make install`)
will include the required dependencies for pre-commit hooks and for building
documentation. Do not use `uv sync --all-extras`: the CUDA variants and some model
extras are mutually exclusive.
To install `pre-commit` in an existing environment, follow the below steps inside the ALCHEMI
Toolkit repository folder:

```bash
pip install pre-commit
pre-commit install
```

Once the above commands are executed, the pre-commit hooks will be activated and all
the commits will be checked for appropriate formatting.

### Continuous Integration (CI)

To ensure quality of the code, your merge request (MR) will pass through several CI checks.
It is mandatory for your MRs to pass these pipelines to ensure a successful merge.
Please keep checking this document for the latest guidelines on pushing code. Currently,
The pipeline has following stages:

1. `format`
    *Pre-commit will check this for you!* Checks for formatting of your
    Python code, using `ruff format` via [Ruff](https://docs.astral.sh/ruff/).
    If your MR fails this test, run `ruff format <script-name>.py` on
    problematic scripts and Ruff will take care of the rest.

2. `lint`
    *Pre-commit will check this for you!*
    Linters will perform static analysis to check the style, complexity, errors
    and more. For markdown files `markdownlint` is used, its suggested to use
    the vscode, neovim or sublime
    [extensions](https://github.com/DavidAnson/markdownlint#related).
    ALCHEMI Toolkit uses `ruff check` via [Ruff](https://docs.astral.sh/ruff/) for
    linting of various types. Currently we use flake8/pycodestyle (`E`),
    Pyflakes (`F`), flake8-bandit (`S`), isort (`I`), and performance 'PERF'
    rules. Many rule violations will be automatically fixed by Ruff; others may
    require manual changes.

3. `license`
    Checks for correct license headers of all files.
    To run this locally use `make license`.

4. `pytest`
    Checks if the test scripts from the `test` folder run and produce desired outputs. It
    is imperative that your changes don't break the existing tests. If your MR fails this
    test, you will have to review your changes and fix the issues.
    To run pytest locally you can simply run `pytest` inside the `test` folder.

5. `coverage`
    Checks if your code additions have sufficient coverage.
    Refer [coverage](https://coverage.readthedocs.io/en/6.5.0/index.html#) for more details.
    If your MR fails this test, this means that you have not added enough tests to the `test`
    folder for your module/functions.
    Add extensive test scripts to cover different
    branches and lines of your additions.
    Aim for more than 80% code coverage.
    To test coverage locally, run the `get_coverage.sh` script from the `test` folder and
    check the coverage of the module that you added/edited.

## Stacking Pull Requests

This repository supports PR stacking: splitting a large change into a series of
small pull requests that build on each other, so every layer can be reviewed on
its own.

GitHub does not currently support dependent pull requests across forks, so all
branches of a stack must be created on this repository rather than on a fork.
This means stacking requires write access to the repository.

Name the branches of a stack `<github handle>/<shared-stack-topic>/<layer>`,
for example `octocat/batched-dynamics/01` and `octocat/batched-dynamics/02`.
The shared topic shows which branches belong to one stack, and the layer is
the position in it.

There are two ways to build a stack, both documented by GitHub:

- With the `gh stack` commands in GitHub CLI, provided by the
  `github/gh-stack` extension (see the [quickstart][gh-stacked-quickstart]):

  ```bash
  gh extension install github/gh-stack  # if `gh stack` is not available
  gh stack init  # first branch, e.g. octocat/batched-dynamics/01
  gh stack add octocat/batched-dynamics/02
  gh stack submit  # push branches and open the linked pull requests
  gh stack sync --prune  # after the bottom pull request merges and is deleted
  ```

- On the GitHub website, without the CLI: open each pull request with its
  base branch set to the layer below, then link the pull requests with the
  **Create stack** option; after merges, the **Rebase stack** button in the
  merge box rebases the remaining layers server-side (see
  [creating stacked pull requests][gh-stacked-web]).

Either way, the first pull request targets `main` and each later one targets
the branch below it, so reviewers see only that layer's diff. When the bottom
pull request merges and its branch is deleted, GitHub retargets the next layer
onto the merged pull request's base branch.

Commits in a stack follow the same rules as any other contribution: sign off
with `git commit -s` and keep pre-commit green.

[gh-stacked-quickstart]: https://docs.github.com/en/pull-requests/get-started/stacked-prs-quickstart
[gh-stacked-web]: https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-stacked-pull-requests
