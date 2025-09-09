# Assignment-1 (AIL7022)
### ~ Kunal Kumar Sahoo

Code Structure:
```
.
├── A1.pdf
├── Q1
│   ├── __pycache__
│   ├── assets
│   ├── env.py
│   ├── output_seeds
│   │   ├── policy_iteration_gamma=0.95.gif
│   │   ├── pvi.gif
│   │   ├── stationary.gif
│   │   ├── tdvi_policy.gif
│   │   ├── time_dependent.gif
│   │   ├── value_iteration_gamma=0.95.gif
│   │   └── vi_stationary_policy.gif
│   ├── part-1
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── main.py
│   │   └── stationary_football_env.csv
│   ├── part-2
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── main.py
│   │   └── non_stationary_football_env.csv
│   ├── part-3
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── main.py
│   │   └── modified_vi_expt.csv
│   └── utils.py
├── Q2
│   ├── or_gym
│   ├── part-1
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── logs
│   │   ├── main.py
│   │   └── output_seeds
│   └── part-2
│       ├── __pycache__
│       ├── main.py
│       ├── plots
│       └── results
├── README.md
└── report.pdf
```

To run any code, go to the corresponding question directory and execute the code as follows:
```bash
python3 -m <module_name>.<code_file_without_extension>
```

For example you want to run the code at `Q1/part-1/main.py`, you can simply execute:
```bash
cd Q1/
python3 -m part-1.main
```


**Kindly follow the above guidelines to run the python scripts, otherwise you may face path issues for the gym environments.**