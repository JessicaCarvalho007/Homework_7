### IMPORT ALL FUNCTIONS FROM A MODULE
import sys
sys.path.append('../case_a')
# sys.path.append('this_is_an_absolute_path_to_my_functions_folder')
import my_functions

observation = 40.  # in celsius
converted_observation = my_functions.celsius_to_fahrenheit(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")


### IMPORT A MODULE AND RENAME IT
import my_functions as mf

observation = 40.  # in celsius
converted_observation = mf.celsius_to_fahrenheit(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")


### IMPORT A SPECIFIC FUNCTION FROM A MODULE

from my_functions import celsius_to_fahrenheit

observation = 40.  # in celsius
converted_observation = celsius_to_fahrenheit(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")


### IMPORT A FUNCTION AND RENAME IT
from my_functions import celsius_to_fahrenheit as c_to_f

observation = 40.  # in celsius
converted_observation = c_to_f(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")

