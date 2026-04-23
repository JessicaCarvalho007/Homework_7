### IMPORT ALL FUNCTIONS FROM A MODULE
import folder_with_functions.my_functions

observation = 40.  # in celsius
converted_observation = folder_with_functions.my_functions.celsius_to_fahrenheit(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")


### IMPORT A MODULE AND RENAME IT
import folder_with_functions.my_functions as mf

observation = 40.  # in celsius
converted_observation = mf.celsius_to_fahrenheit(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")


### IMPORT A SPECIFIC FUNCTION FROM A MODULE

from folder_with_functions.my_functions import celsius_to_fahrenheit

observation = 40.  # in celsius
converted_observation = celsius_to_fahrenheit(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")


### IMPORT A FUNCTION AND RENAME IT
from folder_with_functions.my_functions import celsius_to_fahrenheit as c_to_f

observation = 40.  # in celsius
converted_observation = c_to_f(observation)
print(f"{observation} degrees Celsius is equal to {converted_observation} degrees Fahrenheit.")

