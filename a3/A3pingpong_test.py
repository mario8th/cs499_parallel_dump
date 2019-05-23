

#script to check the output of the ping pong with locs/cond vars program
#may need to change the path to your python bin as necessary
#which python on command line
#you may get a permission denied error when trying to execute:
#use chmod as necessary, e.g., chmod 755 A1sequence2_test.py


import sys


# Get all lines from stdin
lines = sys.stdin.readlines()

total_inc=0
total_dec=0
numerrors=0





for line in lines:
	if "Count is now (inc fn):" in line:
		total_inc+=1
	if "Count is now (dec fn):" in line:
		total_dec+=1
	


print("Testing output...")
print("Total increments (should be 3x10): "+str(total_inc)) 
print("Total decrements (should be 2x10): "+str(total_dec)) 	



#possible errors:
if (total_inc!=30):
	print("Error: you should have 30 increments in total printed")
	numerrors+=1
if (total_dec!=20):
	print("Error: you should have 20 decrements in total printed")
	numerrors+=1


print("Total errors detected: "+ str(numerrors))



