# tutorial: collaborative ml model improvement

in this tutorial students will work together to improve a baseline model by experimenting with different hyperparameters and configurations.



### setup
1. clone this repository
2. create and activate virtual environment:
   ```bash
   python -m venv ml_tutorial_env
   source ml_tutorial_env/bin/activate  # on windows: ml_tutorial_env\Scripts\activate
   ```
3. install dependencies: `pip install -r requirements.txt`
4. run baseline to verify working setup without logging results: `python train.py --no-log` 

### experimentation 
1. **create a feature branch**: `git checkout -b [experiment-name]/[your initials]`
2. **create your config file**: copy `config.json` to `config_[experiment-name].json` and modify it
3. **train your model**: run `python train.py --config config_[experiment-name].json`
4. **document results**: your results are automatically logged to `evaluation/results_{experiment-name}.json`
5. **update evaluation summary**: add a brief description to `evaluation/summary.md`
6. **commit changes**: use descriptive commit messages
7. **push and create pr**: submit pull request with detailed description




## baseline performance
- Accuracy: ~0.83
- F1-score: ~0.82


## reference for git commands
```bash
# create and switch to feature branch
git checkout -b experiment-name/initials

# stage and commit changes
git add .
git commit -m "feat: experiment with [description]"

# push branch and create pr (on github)
git push origin experiment-name/initials

# switch back to main
git checkout main

# pull latest changes
git pull origin main
```

## project Structure
```
├── README.md                 # This file
├── config_baseline.json              # Baseline model hyperparameters
├── config_[name].json        # Individual student configs (no conflicts!)
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
└── evaluation/
    ├── results.json          # Experiment results log
    └── README.md            # Summary of all experiments
```



## notes:
- **no merge conflicts**: each student uses their own `config_[experiment-name].json` file
- **automatic logging**: results are automatically added to `evaluation/results_[experiment-name].json`  
- **clean environment**: virtual environment keeps dependencies isolated