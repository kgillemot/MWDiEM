def array_min_max_finder(x):
	xmax=x[0]
	xmin=x[0]
	indmax=0
	indmin=0
	for i in range(0,len(x)):
		if x[i]<xmin:
			xmin=x[i]
			indmin=i
		if x[i]>xmax:
			xmax=x[i]
			indmax=i
	return (xmin,xmax,indmin,indmax)
