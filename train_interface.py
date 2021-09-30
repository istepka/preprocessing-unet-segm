from numpy.core.records import array
from train import Trainer
import json, csv
import numpy as np


def run_training_from_string(run_string) -> array:
    '''Interface for running training just by passing config string. \n 
    e.g. \"Bs+Aug+Heq+Pcn\"\n
    Return: `run_metrics`'''

    print('-------------------------------------------------------------')
    print(f'Run name: {run_string}')
    print('-------------------------------------------------------------')


    trainer = Trainer() 
    trainer.mlflow_run_logs(run_name=run_string)
    trainer.load_prep_from_settings_string(run_string)
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.evaluate()
    run_metrics = trainer.test_model_additional_metrics()
    trainer.save()
    trainer.mlflow_stop_logs()

        
    print('Run has been executed succesfully.')
    return run_metrics

def run_training_from_config(config_file='config/runs.json') -> str:
    '''
    Run training from config file. It will generate .csv file with key metrics.\n
    Return: `filepath` to csv file containing summary  
    '''

    with open(config_file, 'r') as file:
        runs_data = json.load(file)

    with open(runs_data['out_csv_name'],'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Accuracy', 'Jaccard index', 'Precision', 'Sensitivity', 'Specifitivity'])
    
    for data_dir in runs_data['directiories']:
        for i, run in enumerate(runs_data['runs']):
            print('-'*25)
            print(f'Run name: {run["run_name"]}')
            print('-'*25)

            repeats = runs_data["repeats"]
            metrics = np.array([0.,0.,0.,0.,0.], dtype=np.float)
            deviation = np.zeros((repeats,5), dtype='f')
            

            for rep in range(repeats):
                print('-'*25)
                print(f'Rep: {rep}')
                print('-'*25)

                trainer = Trainer(
                epochs=runs_data['epochs'],
                data_directory=data_dir,
                preprocessing_params=run['prep_params'],
                batch_size=runs_data['batch_size']
                )
            
                trainer.mlflow_run_logs(run_name=run["run_name"])
                trainer.load_data()
                trainer.build_model()
                trainer.train()
                trainer.evaluate()
                run_metrics = trainer.test_model_additional_metrics()
                trainer.save()
                trainer.mlflow_stop_logs()

                metrics = np.add(metrics, run_metrics)
                deviation[rep] = run_metrics

                del trainer
            
            metrics /= repeats
            stdev = np.std(deviation, axis=0)

            print('-'*25)
            print(f'Averaged metrics for {run["run_name"]}: {metrics}')
            print(f'Standard deviation for {run["run_name"]}: {stdev}')
            print('-'*25)

            #Save data after each run
            with open(runs_data['out_csv_name'],'a') as f:
                writer = csv.writer(f)

                row_data = list()
                for metric, std in zip(metrics, stdev):
                    row_data.append(f'{metric} +- {std}')
                
                writer.writerow( [run["run_name"], *row_data])

    print(f'All runs have been done and results can be found in: {runs_data["out_csv_name"]}.')
    return runs_data['out_csv_name']


if __name__ == '__main__':
    run_training_from_config()