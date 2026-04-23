import sys
sys.path.append('/workspaces/hwrs564b_course_materials_sandoval/content_weekly/Week13_Workflows/session2_activity2/case_a')
import my_functions

my_R = 1
my_gradient =0.1
my_K = 1

Q = my_functions.compute_darcy(my_K, my_R, my_gradient)

print(Q)