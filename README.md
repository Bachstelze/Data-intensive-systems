---
title: Github Sync Space
emoji: 🔥
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.47.0
app_file: app.py
pinned: false
---
# Data-intensive-systems
Group project to develop in a machine learning setting. The developed [gradio](https://www.gradio.app) app for movement assessment is automatically deployed to the [synchronized huggingface space](https://huggingface.co/spaces/Bachstelze/github_sync).
## Members

1. Reem
2. Rasa
3. Amol
4. Kalle


## Usage

Installation of [gardio](https://www.gradio.app/docs/gradio/interface) to build machine learning apps in Python:
```pip install gradio```

Run the app with ```python3 app.py```

### Week 1 - Kalle team lead, rest developers

- Slack as a medium of communications
- GitHub for version control
- This GitHub issue tracker as a simple project management tool
- Synchronization of this GitHub repo with the [hugginface space](https://huggingface.co/spaces/Bachstelze/github_sync)
- Local development, switch to [cscloud](https://cscloud.lnu.se/project/) if needed.
- Meetings as and when needed

### Week 2 - first linear regression model development and deployment

- Testing pickled models with test/test_model.py

### Week 4 - refactor codebase

- add gitattributes for LSF usage to handle pickle files. This is possible for more file types, e.g. ```git lfs track "*.pdf" + git add .gitattributes```
- Update numpy version requirement, older versions can't load pickle files from new versions.
- The requirement for scikit-learn is updated to 1.8.0 to prevent InconsistentVersionWarning.
- Refactor and lint app.py

### Outlook
Input the raw video into a [reasoning vision model](https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning) and develop the workflow with [daggr](https://huggingface.co/blog/daggr) for gradio.


### Development & Contribution Guidelines

Currently, the team commits directly to the `main` branch.  
Instead of formal pull request restrictions, we follow shared development and quality guidelines to ensure stability, especially for the HuggingFace deployment and ML workflows.

These rules will also prepare the project for a future PR-based workflow.

---

#### General development rules

- Follow the existing project structure and naming conventions
- Keep notebooks and scripts readable and well-documented
- Avoid committing temporary, experimental, or unused code
- Document design decisions when introducing new components

---

#### HuggingFace deployment safety

- Do NOT commit `.pdf` or `.xlsx` files (they break HuggingFace sync)
- Avoid committing large model artifacts unless required
- Ensure changes do not break the deployment app (`app.py`)

---

#### Code quality expectations

Before pushing to `main`:

- Run local linting checks when possible
- Ensure notebooks execute without errors
- Remove unused imports and variables
- Keep code modular and readable

CI/CD will enforce:

- Linting checks for `.py` and `.ipynb`
- Automated tests 
- Deployment only after quality checks pass

---

#### Design and coding principles

- Write clear, modular Python code
- Separate experimentation (notebooks) from production logic
- Reusable functions should live in `.py` modules
- Use consistent naming and documentation practices

---

#### Team review practice

Even without PRs, team members:

- Communicate major changes in Slack/GitHub Issues
- Request peer feedback before large commits
- Document model changes and performance improvements
- Flag anything that may affect deployment

---

#### Future transition

As the project grows, we may transition to:

- Feature branches
- Pull request reviews
- Mandatory CI approval before merge

These guidelines are the first step toward a structured DevOps workflow.

