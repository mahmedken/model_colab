# experiment results summary

this file tracks all experiments and their evaluation. each student should add a brief summary after running their experiments.

## instructions to add results

after running your experiment, add a new section below with:
1. **your name** and **experiment name**
2. **brief description** of what you changed
3. **results** (accuracy, f1-score)

---

## experiment Log

### kenny - baseline
- **config**: 2 layers [8,4], relu, dropout=0.2, lr=0.001, batch size=16, epochs=100
- **results**: accuracy=0.8667, f1=0.8611
- **description**: initial baseline configuration for comparison

---

# copy this section for each new experiment -- don't edit this template
### [your Name] - [experiment name]  
- **config**: [describe your configuration changes]
- **results**: accuracy=x.xxx, f1=x.xxx
- **description**: [what did you try?]

Sultana Yeasmin
Experiment-fun

I simply doubled the number of epochs and batch size.

I recieved a baseline accuracy and f1 of about .96.
But with this new change, I received a accuracy and f1 of about .93.
Then, I kept redoing the command and retrained the model multiple times. 
With this, the accuracy and f1 changed to .8, then about .7 ish.
My hypothesis is that this is due to overfitting and that the model had been exposed to training data from previous runs. 
