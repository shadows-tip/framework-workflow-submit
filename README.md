# A novel framework to accelerate the elimination of neglected tropical diseases
The objective of this study is to propose a novel, One Health-driven framework to guide and accelerate the elimination of NTDs.

### Environmental Dependencies
This project requires Python 3.9 or newer. Here are the main dependencies:
numpy==1.22.0  
pyDOE2==1.3.0  
scipy==1.7.3  
joblib==1.3.1  
plotnine==0.10.1

### Usage
Executing the following code on the terminal will automatically call the propagation dynamics model to generate a dataset. During this process, a folder will be automatically created and the dataset will be automatically saved to the corresponding folder.  
```python
python DataGenerator.py
 ```
Executing the above code will generate the dataset required for machine learning, and executing the following code at the terminal will use the dataset for machine learning. The training process is automated until the end of execution.  
```python
python MachineLearning.py
```

