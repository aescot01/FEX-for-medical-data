Here is a mod of the fex_local_approximation code which allows us to learn infection rate of covid as a function of certain land density data. Currently setup with data for Autuga county, Alabama between 10/1/2020 and 4/1/2021.

steps:
1) copy 'controller2.py' and  useable_data into the standard 'fex_local_approximation' code
2) in 'scripts.py', replace all instances of 'controller.py' with 'controller2.py'
3) make necessary changes to controller2.py so that it is using the data provided in 'useable data'
