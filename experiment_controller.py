from sklearn import preprocessing
from train_gpu import Trainer
import json
import numpy as np
import csv

bacteria_dir = 'npy_datasets/bacteria/'
lesion_dir = 'npy_datasets/cv_data/'
resolution = 256
repeats = 3
epochs = 100
runs = []

with open('runs.json') as f:
    runs = json.load(f)

with open('benchmark.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Accuracy', 'Jaccard index', 'Precision', 'Sensitivity', 'Specifitivity'])


for dataset_directiory in [lesion_dir, bacteria_dir]:
    for run_dict in runs:

        metrics = np.array([0.,0.,0.,0.,0.], dtype=np.float)
        
        if 'bacteria' in dataset_directiory:
            suffix = ' - bacteria'
        else:
             suffix = ' - lesion'

        run_name = run_dict['run_name'] + suffix
        prep_params = run_dict['prep_params']

        print()
        print('-'*25)
        print(f'RUN: {run_name}, PARAMS: {prep_params}')
        print('-'*25)
        print()

        for i in range(repeats):
            print()
            print('-'*25)
            print(f'Rep: {i}')
            print('-'*25)
            print()

            trainer = Trainer(
                epochs=epochs,
                data_directory=dataset_directiory,
                preprocessing_params=prep_params
                )

            trainer.mlflow_run_logs(run_name=run_name)
            trainer.load_data()
            trainer.build_model()
            trainer.train()
            trainer.evaluate()
            run_metrics = trainer.test_model_additional_metrics()
            trainer.save()
            trainer.mlflow_stop_logs()

            print(metrics, run_metrics)
            metrics = np.add(metrics, run_metrics)
            del trainer

        metrics /=repeats
        print()
        print('-'*25)
        print(f'Averaged metrics for {run_name}: {metrics}')
        print('-'*25)
        print()

        with open('benchmark.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow( [run_name, *metrics])
        



