# Contribution workflow

This page gives a command-line workflow for contributors who have write access
to the repository. If you do not have write access, fork the repository first
and push your branch to your fork; the remaining steps are the same.

If Git is new to you, GitHub's
[Git Handbook](https://docs.github.com/en/get-started/using-git/about-git) explains
the basic ideas and terminology.

## 1. Create or choose an issue

Before starting, search the [existing issues](https://github.com/JeffersonLab/nps-sro-ml/issues)
and pull requests to avoid duplicating work.

[Create an issue](https://github.com/JeffersonLab/nps-sro-ml/issues/new) when you
want to report a bug, request a feature, or propose a substantial change. Include:

- what you expected to happen and what happened instead;
- a small reproducible example, command, or error message, when applicable;
- the software version, computing environment, and relevant input data;
- plots or logs that help explain the problem, with sensitive information
  removed; and
- the scientific impact, especially if results or physics definitions may
  change.

A separate issue is optional for a small typo or an obvious, self-contained
fix. For a large change, discuss the design in an issue before investing a lot
of time in implementation.

## 2. Update your local `main` branch

Clone the repository once:

```bash
git clone https://github.com/JeffersonLab/nps-sro-ml.git
cd nps-sro-ml
```

Before starting each contribution, update `main`:

```bash
git switch main
git pull --ff-only origin main
```

`--ff-only` prevents Git from creating an accidental merge commit while you
are updating your local branch.

## 3. Create a branch

Create every working branch from `main`:

```bash
git switch -c issue-42-fix-batch-indexing
```

Use a short, descriptive name such as `docs/add-training-guide`,
`fix/empty-event`, or `issue-42-fix-batch-indexing`. Do not make contributions
directly on `main`.

## 4. Make and inspect the change

Work in small steps. Check which files have changed and inspect the diff:

```bash
git status
git diff
```

Run the relevant formatting and test commands described in
[Contributing code](./code.md). Then stage and commit the intended files:

```bash
git add path/to/changed-file path/to/test-file
git diff --staged
git commit -m "Fix batch indexing for empty events"
```

Write commit messages in the imperative mood and say what the change does.
Avoid `git add .` until you are comfortable checking that it did not stage
data, credentials, build products, or unrelated work.

## 5. Push and open a pull request

Push the branch to GitHub:

```bash
git push -u origin issue-42-fix-batch-indexing
```

Open the link printed by Git, or go to the repository's
[pull requests](https://github.com/JeffersonLab/nps-sro-ml/pulls) page. Select
your branch as the **compare** branch and `main` as the **base** branch.

In the PR description:

- explain the problem and the solution in plain language;
- link the issue with `Closes #42` when the PR fully resolves it;
- list the commands you ran to test the change;
- describe changes to data selection, units, labels, random seeds, model
  configuration, or physics assumptions;
- include before-and-after plots for changes that affect numerical or visual
  results; and
- call out known limitations or follow-up work.

Opening a **draft pull request** is encouraged when you want early feedback.

## 6. Respond to review

Review is a conversation. Ask when a comment is unclear. Make requested changes
on the same branch, commit them, and push again; the PR updates automatically.

If `main` changes while your PR is open, update your branch:

```bash
git switch main
git pull --ff-only origin main
git switch issue-42-fix-batch-indexing
git merge main
```

Resolve any conflicts carefully, rerun the relevant tests, and push. Ask for
help before guessing about a conflict in unfamiliar code.

## 7. Merge and clean up

Merge only after review is complete and automated checks pass. Delete the
remote branch using GitHub's **Delete branch** button. Then clean up locally:

```bash
git switch main
git pull --ff-only origin main
git branch -d issue-42-fix-batch-indexing
```

Your contribution is now part of the project.

