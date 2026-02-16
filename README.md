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


### PR Review Guidelines

To ensure code quality, stability of the HuggingFace deployment, and consistent development practices, all pull requests must follow these guidelines:

#### Before creating a PR
- Code must follow project structure and naming conventions  
- No `.pdf` or `.xlsx` files should be committed (these break HuggingFace sync)  
- Code should pass local linting checks  
- Changes should be tested locally when possible  

#### CI/CD requirements
- Linting must pass before merge  
- CI pipeline must complete successfully  

#### PR description must include
- Clear explanation of what was implemented or changed  
- Reason for the change  
- Screenshots or demo output (if UI/model related)  

#### Reviewer checklist
- Code readability and structure  
- Consistency with project design  
- No restricted file uploads (.pdf/.xlsx)  
- CI pipeline passes  
- No breaking changes to deployment  

#### Merge rules
- PR cannot be merged if CI fails  
- PR must be up-to-date with `main` branch before merging

