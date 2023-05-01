import dataload

import datetime

# Get the current time
start = datetime.datetime.now()

# Print the current time
print("start time", start)
data = dataload.dmg777k()
end = datetime.datetime.now()
print("end_time", end)
print("elapsed:",end-start)
#print(data.i2e)