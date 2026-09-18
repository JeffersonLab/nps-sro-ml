# 🐛 Reporting a bug

A useful bug report helps us reproduce the problem and understand its impact.

You do not need to diagnose the cause before reporting it.

## 🔎 Before opening an issue

1. Search the [existing issues](https://github.com/JeffersonLab/nps-sro-ml/issues) for the error message or behavior you observed.
2. If an open issue describes the same problem, add any new information there.
3. If a similar issue is closed but the problem still occurs, open a new issue and link to the closed one.

## 📝 Open a bug report

[Create a GitHub issue](https://github.com/JeffersonLab/nps-sro-ml/issues/new) with a short, descriptive title.

Include the following information when it is relevant:

* 🐛 **What happened:** Describe the incorrect behavior or result.
* 🎯 **What you expected:** Explain what should have happened instead.
* 🔁 **How to reproduce it:** List the commands and steps that trigger the problem. A small example is ideal.
* 💥 **Error output:** Copy the complete error message or traceback into a Markdown code block.
* 🖥️ **Environment:** Include your operating system, repository commit or branch, and relevant Python, PyTorch, CUDA, ROOT, or LibTorch versions.
* 🔬 **Scientific context:** Note the dataset, selection, units, model configuration, and random seed when they affect the problem.
* 📎 **Evidence:** Attach a small plot, screenshot, or log when it makes the issue easier to understand.

You can copy this template into the issue:

```markdown
## 🐛 What happened?

## 🎯 What did you expect?

## 🔁 Steps to reproduce

1.
2.
3.

## 💥 Error message or unexpected output

## 🖥️ Environment

- Commit or branch:
- Operating system:
- Python and PyTorch versions:
- CUDA, ROOT, or LibTorch versions, if relevant:

## 🔬 Additional scientific context

Dataset, selections, units, model configuration, random seed, or plots.
```

## 🔐 Protect data and credentials

Do not attach restricted detector data, access tokens, passwords, private keys, or other sensitive information.

Prefer a small synthetic example when the original input cannot be shared. If the problem requires restricted data to reproduce, describe the data and its location without uploading it.

Intermittent problems are still worth reporting. State how often the problem occurs and any conditions that make it more or less likely.
