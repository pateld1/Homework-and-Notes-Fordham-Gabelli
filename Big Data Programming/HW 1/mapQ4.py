#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:

	date = line[6:14]
	hotTemp = float(line[39:45])
	coldTemp = float(line[47:53])

	if hotTemp > 40:
		print('%s\t%s' % ('Hot Day', date))
	if coldTemp < 10:
		print('%s\t%s' % ('Cold Day', date))



