#!/usr/bin/env python

from operator import itemgetter
import sys

# input comes from STDIN
for line in sys.stdin:

	line = line.strip()

	day, date = line.split('\t', 1)

	print('%s\t%s' % (day, date))
