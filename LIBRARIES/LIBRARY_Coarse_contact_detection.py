import numpy as np
import operator
from collections import defaultdict



#####mostmar elvben nem hasznalom
def Particles_to_cells(max_particlesize,numpart,particle,sizebox):
	numcells=int(sizebox/max_particlesize)
	cellmax=sizebox
	cellsize=cellmax/numcells
	cell=[[[[] for _ in range(numcells)] for _ in range(numcells)] for _ in range(numcells)]
	#xcell=[]
	#ycell=[]
	#zcell=[]
	#partnumcell=[]

	for i in range(0,numpart):
		cm=particle[i].center_of_mass
		
		xc=int(cm[0]/cellsize)
		yc=int(cm[1]/cellsize)
		zc=int(cm[2]/cellsize)
		cell[xc][yc][zc].append(i)
		#xcell.append(xc+0.5)
		#ycell.append(yc+0.5)
		#zcell.append(zc+0.5)
		#partnumcell.append(i)	
	#func.Plot_particles(xcell,ycell,zcell,partnumcell)
	return(cell)




	
#not in use any more
def Linked_cell(cell):
	neigbourschecked=1
	pairstocheck=[]
	for i in range(0,len(cell)):
		for j in range(0,len(cell)):
			for k in range(0,len(cell)):
				#egyaltalan kell e itt part keresni
				if np.matrix(cell[i][j][k]).shape[1]>0:
					#az az eset, amikor a sajat cellajaban is kell part keresni
					if np.matrix(cell[i][j][k]).shape[1]>1:
						#print("egy cellaban levok")
						for m in range(0,np.matrix(cell[i][j][k]).shape[1]-1):
							for n in range(m+1,np.matrix(cell[i][j][k]).shape[1]):
								#print(str(cell[i][j][k][m])+" es "+str(cell[i][j][k][n])+" szomszedok")
								pairstocheck.append([cell[i][j][k][m],cell[i][j][k][n]])
					#print("kulon cellaban levok")
					#es most keresunk parokat intercellaban
					for l in range(-1,2):
						if i+l<=len(cell)-1 and i+l>=0:
							for m in range(-1,2):
								if j+m<=len(cell)-1 and j+m>=0:
									n=1
									if k+n<=len(cell)-1:
										for p in range(0,np.matrix(cell[i][j][k]).shape[1]):
											for q in range(0,np.matrix(cell[i+l][j+m][k+n]).shape[1]):
												#print(str(cell[i][j][k][p])+" es "+str(cell[i+l][j+m][k+n][q])+" szomszedok")
												pairstocheck.append([cell[i][j][k][p],cell[i+l][j+m][k+n][q]])
					for l in range(-1,2):	
						if i+l<=len(cell)-1 and i+l>=0:
							m=1
							n=0
							if j+m<=len(cell)-1:
								for p in range(0,np.matrix(cell[i][j][k]).shape[1]):
									for q in range(0,np.matrix(cell[i+l][j+m][k+n]).shape[1]):
										#print(str(cell[i][j][k][p])+" es "+str(cell[i+l][j+m][k+n][q])+" szomszedok")	
										pairstocheck.append([cell[i][j][k][p],cell[i+l][j+m][k+n][q]])		

					l=1
					m=0
					n=0
					if i+l<=len(cell)-1:
						for p in range(0,np.matrix(cell[i][j][k]).shape[1]):
							for q in range(0,np.matrix(cell[i+l][j+m][k+n]).shape[1]):
								#print(str(cell[i][j][k][p])+" es "+str(cell[i+l][j+m][k+n][q])+" szomszedok")	
								pairstocheck.append([cell[i][j][k][p],cell[i+l][j+m][k+n][q]])					
	return(pairstocheck)
	
def Linked_cell_hash(particle,numcells_x,numcells_y,num_part_min,num_part_max,pairstocheck): #,wall_dict):
	hashdict={}
	for i in range(num_part_min,num_part_max):
		hashdict[particle[i].particle_id] =particle[i].hashh

	rev_multidict = {}
	for key, value in hashdict.items():
		rev_multidict.setdefault(value, set()).add(key)
	
	#def Merge(dict1, dict2): 
	#	return(dict1.update(dict2)) 
	#Merge(rev_multidict, wall_dict)

	#print("kata",rev_multidict)
	#pairstocheck=[]
	
	for key in rev_multidict:
		if len(rev_multidict[key])>1:
			for j in range(0,len(rev_multidict[key])):
				for k in range(j+1,len(rev_multidict[key])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key])[k]])
			
		if key+1 in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+1])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+1])[k]])
		if key+numcells_x in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+numcells_x])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+numcells_x])[k]])
		if key+numcells_x*numcells_y in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+numcells_x*numcells_y])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+numcells_x*numcells_y])[k]])
	
		if key+numcells_x*numcells_y+1 in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+numcells_x*numcells_y+1])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+numcells_x*numcells_y+1])[k]])
		if key+numcells_x*numcells_y+1+numcells_x in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+numcells_x*numcells_y+1+numcells_x])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+numcells_x*numcells_y+1+numcells_x])[k]])
		if key+numcells_x*numcells_y+numcells_x in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+numcells_x*numcells_y+numcells_x])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+numcells_x*numcells_y+numcells_x])[k]])
		if key+numcells_x+1 in rev_multidict:
			for j in range(0,len(rev_multidict[key])):
				for k in range(0,len(rev_multidict[key+numcells_x+1])):
					pairstocheck.append([list(rev_multidict[key])[j],list(rev_multidict[key+numcells_x+1])[k]])
	#for i in range(0,len(rev_multidict)):
		#if len(rev_multidict[i])>1:
		#	for j in range(0,len(rev_multidict[i])):
		#		for k in range(j+1,len(rev_multidict[i])):
		#			pairstocheck.append([list(rev_multidict[i])[j],list(rev_multidict[i])[k]])
		##print(pairstocheck)

		#if i+1 in rev_multidict:
			#for j in range(0,len(rev_multidict[i])):
				#for k in range(0,len(rev_multidict[i+1])):
					#pairstocheck.append([list(rev_multidict[i])[j],list(rev_multidict[i+1])[k]])
		##print(pairstocheck)
					
		#if i+numcells in rev_multidict:
			#for j in range(0,len(rev_multidict[i])):
				#for k in range(0,len(rev_multidict[i+numcells])):
					#pairstocheck.append([list(rev_multidict[i])[j],list(rev_multidict[i+numcells])[k]])
		##print(pairstocheck)
			
		#if i+numcells**2 in rev_multidict:
			#for j in range(0,len(rev_multidict[i])):
				#for k in range(0,len(rev_multidict[i+numcells**2])):
					#pairstocheck.append([list(rev_multidict[i])[j],list(rev_multidict[i+numcells**2])[k]])
		##print(pairstocheck)
					
	return(pairstocheck)
	
#def Wall_hash(particle,num_part_max):		
	#wall_dict={}
	#for i in range(0,num_part_max):
		#if particle[i].typee==0:
			#for j in range(0,len(particle[i].hashh)):
				#wall_dict.setdefault(particle[i].hashh[j], [])
				#wall_dict[particle[i].hashh[j]].append(i)
	
	#return(wall_dict)


	
