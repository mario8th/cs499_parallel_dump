# Script to check the output of the housekeeper simulation 
#
import sys
import re
from operator import eq

# Helper function
def get_thread_name(line):
	p = re.compile('\[.+\]');
	match = p.match(line)
	if (match == None):
		return None
	name = match.group().replace('[','').replace(']','')
	words = name.split(" ");
	#print "WORDS=",words
	if ((not words[0] == "Washer/dryer") and  (not words[0] == "Washer") and (not words[0] == "Dryer")) or (not words[1] == "housekeeper") or (not words[2].isdigit()):
		return None
	else:
		return name

def get_all_thread_names(lines):
	names = []
	for linenum in xrange(0,len(lines)):
  		name = get_thread_name(lines[linenum])
		if name is None:
			print "Invalid/missing thread name at line ",linenum
			print "\t --> "+lines[linenum]+"\n"
			exit(1)
		else:
			if name not in names:
				names.append(name)
	return names


def get_num_threads(lines):
	names = get_all_thread_names(lines)
	split_names = []
	for name in names:
		split_names.append(name.split(" "))

	washers =  sorted([int(z) for (x,y,z) in split_names if x == "Washer"])
	dryers = sorted([int(z) for (x,y,z) in split_names if x == "Dryer"])
	both =      sorted([int(z) for (x,y,z) in split_names if x == "Washer/dryer"])

	if (washers != range(0,max(washers)+1)) or (dryers != range(0,max(dryers)+1)) or (both != range(0,max(both)+1)): 
		print "Invalid thread ids!!"
		print "\tWasher ids: ",washers
		print "\tDryer ids: ",dryers
		print "\tBoth washer/dryer ids: ",both
		exit(1)

	return (len(washers), len(dryers), len(both))



#########################################################

# Constants
num_iterations = 10
num_washers = 3
num_dryers = 3
	
# Array of thread names
threadids = []

# Get all lines from stdin
lines = sys.stdin.readlines()

# Get number of threads of each kind
print "Checking that the output is well-formatted..."
(num_washers_housekeepers, num_dryers_housekeepers, num_washer_dryer_housekeepers) = get_num_threads(lines)
print "\tDetected", num_washers_housekeepers, "washer housekeepers"
print "\tDetected", num_dryers_housekeepers, "dryer housekeepers"
print "\tDetected", num_washer_dryer_housekeepers, "both washer and dryer housekeepers"

###
### Check that every thread does every activity 10 times and then leaves
###

print "Checking that every housekeeper does its required number of operations..."
# declare dictionaries
working = {}
waiting = {}
articles_cleaned = {}
washer_wanting = {}
dryer_wanting = {}
washer_getting = {}
dryer_getting = {}
washer_putting_back = {}
dryer_putting_back = {}

# initialize dictionaries
for line in lines:
	name = get_thread_name(line)
	working[name] = 0
	waiting[name] = 0
	articles_cleaned[name] = 0
	washer_wanting[name] = 0
	dryer_wanting[name] = 0
	washer_getting[name] = 0
	dryer_getting[name] = 0
	washer_putting_back[name] = 0
	dryer_putting_back[name] = 0

# populate dictionaries
for line in lines:
	name = get_thread_name(line)
	if "working" in line:
		working[name] += 1
	elif "waiting" in line:
		waiting[name] += 1
	elif "has taken articles out of the" in line:
		articles_cleaned[name] += 1
	elif "washer" in line and "wants" in line:
		washer_wanting[name] += 1
	elif "dryer" in line and "wants" in line:
		dryer_wanting[name] += 1
	elif "washer" in line and "has got" in line:
		washer_getting[name] += 1
	elif "dryer" in line and "has got" in line:
		dryer_getting[name] += 1
	elif "washer" in line and "has finished with" in line:
		washer_putting_back[name] += 1
	elif "dryer" in line and "has finished with" in line:
		dryer_putting_back[name] += 1

# check validity
for name in get_all_thread_names(lines):
	if working[name] != num_iterations:
		print name+" isn't working "+str(num_iterations)+" times!"
		exit(1)
	if working[name] != num_iterations:
		print name+" isn't waiting "+str(num_iterations)+" times!"
		exit(1)
	if articles_cleaned[name] != num_iterations:
		print name+" hasn't taken articles out "+str(num_iterations)+" times!"
		exit(1)
	housekeeper_type = name.split(" ")[0]
	if housekeeper_type == "Washer housekeeper" or housekeeper_type == "Washer and dryer housekeeper":
		if (washer_wanting[name] != num_iterations):
			print name+" isn't wanting a washer "+str(num_iterations)+" times!"
			exit(1)
		if (washer_getting[name] != num_iterations):
			print name+" isn't getting a washer "+str(num_iterations)+" times!"
			exit(1)
		if (washer_putting_back[name] != num_iterations):
			print name+" isn't finishing with a washer "+str(num_iterations)+" times!"
			exit(1)
	if housekeeper_type == "Dryer housekeeper" or housekeeper_type == "Washer and dryer housekeeper":
		if (dryer_wanting[name] != num_iterations):
			print name+" isn't wanting a dryer "+str(num_iterations)+" times!"
			exit(1)
		if (dryer_getting[name] != num_iterations):
			print name+" isn't getting a dryer "+str(num_iterations)+" times!"
			exit(1)
		if (dryer_putting_back[name] != num_iterations):
			print name+" isn't finishing with a dryer "+str(num_iterations)+" times!"
			exit(1)

print "\tEvery thread does what it needs to do",num_iterations,"times."


###
### Check that more washers or dryers haven't been used than available
###
print "Checking that more washers or dryers haven't been used than available..."

num_taken_washers = 0
num_taken_dryers = 0
for linenum in xrange(0,len(lines)):
	line = lines[linenum]
	if "has got a washer" in line:
		num_taken_washers += 1
	if "has got a dryer" in line:
		num_taken_dryers += 1
	if (num_taken_washers > num_washers):
		print "More than "+str(num_washers)+" washers are taken!! (line "+str(linenum)+")"
		exit(1)

	if (num_taken_dryers > num_dryers):
		print "More than "+str(num_dryers)+" dryers are taken!! (line "+str(linenum)+")"
		exit(1)
	if "has finished with the washer" in line:
		num_taken_washers -= 1
	if "has finished with the dryer" in line:
		num_taken_dryers -= 1
	
print "\tNo more washers or dryers are used than are available."

###
### Check that with many threads, sometimes two are using the washers/dryers concurrently
### 
print "Checking that housekeepers are able to housekeep at the same time..."
num_washer_housekeepers_washing = 0
num_dryer_housekeepers_drying = 0
num_both_type_housekeepers_washing_drying = 0

max_num_washer_housekeepers_washing = 0
max_num_dryer_housekeepers_drying = 0
max_num_both_type_housekeepers_washing_drying = 0

for linenum in xrange(0,len(lines)):
	line = lines[linenum]
	name = get_thread_name(line)

	if "has put laundry in" in line:
		if "Washer housekeeper" in name:
			num_washer_housekeepers_washing += 1
		if "Dryer housekeeper" in name:
			num_dryer_housekeepers_drying += 1
		if "Washer and dryer housekeeper" in name:
			num_both_type_housekeepers_washing_drying += 1

	max_num_washer_housekeepers_washing = max(max_num_washer_housekeepers_washing,num_washer_housekeepers_washing)
	max_num_dryer_housekeepers_drying = max(max_num_dryer_housekeepers_drying,num_dryer_housekeepers_drying)
	max_num_both_type_housekeepers_washing_drying = max(max_num_both_type_housekeepers_washing_drying,num_both_type_housekeepers_washing_drying)

	if "has taken articles out of the" in line:
		if "Washer housekeeper" in name:
			num_washer_housekeepers_washing -= 1
		if "Dryer housekeeper" in name:
			num_dryer_housekeepers_drying -= 1
		if "Washer and dryer housekeeper" in name:
			num_both_type_housekeepers_washing_drying -= 1
		
if max_num_washer_housekeepers_washing == 1 and num_washers_housekeepers > 1:
	print "\tError: no more than 1 washer housekeeper uses washer at a time!"
	exit(1)
if max_num_dryer_housekeepers_drying == 1 and num_dryers_housekeepers > 1:
	print "\tError: no more than 1 dryer housekeeper uses the dryer at a time!"
	exit(1)
if max_num_both_type_housekeepers_washing_drying == 1 and num_washer_dryer_housekeepers > 1:
	print "\tError: no more than 1 washer and dryer housekeeper uses the washer/dryer at a time!"
	exit(1)

print "\tHousekeepers can housekeep concurrently."

print "NO ERRORS DETECTED"
