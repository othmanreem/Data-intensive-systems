import pickle
import os

# Check A6 SVM model
a6_path = './models/champion_svm.pkl'
with open(a6_path, 'rb') as f:
    artifact = pickle.load(f)
    #print(artifact)
    #print(artifact.get('feature_columns'))
print('A6 SVM Model Structure:')
print(f'  Type: {type(artifact)}')
print(f'  Class name: {type(artifact).__name__}')
if hasattr(artifact, 'steps'):
    print(f'  Steps: {[step[0] for step in artifact.steps]}')
    for step_name, step in artifact.steps:
        print(f'    {step_name}: {type(step).__name__}')
        if hasattr(step, 'feature_names_in_'):
            print(f'      feature_names_in_: {step.feature_names_in_}')
        if hasattr(step, 'get_feature_names_out'):
            try:
                fnames = step.get_feature_names_out()
                print(f'      get_feature_names_out(): {fnames}')
            except Exception as e:
                print(f'      get_feature_names_out() error: {e}')
if isinstance(artifact, dict):
    print(f'  Keys: {artifact.keys()}')
    if 'feature_columns' in artifact:
        print(f'  feature_columns: {artifact["feature_columns"]}')
