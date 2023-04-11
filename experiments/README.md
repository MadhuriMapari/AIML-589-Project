# Experiments

In this folder, each experiment is defined in a separate script. These script needs at least one argument giving the environment to use and have to be run at the root of the project :   
```
python -m experiments.<script_name>
```

Details about script argument can be found using `-h` :
 ```
 python -m GPRL.experiments.<script_name.py> -h
 ```

By default, results are saved at : `experiments\results\<experiment_name>`