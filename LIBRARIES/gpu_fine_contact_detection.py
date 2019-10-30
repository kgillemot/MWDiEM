import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import sys
from jinja2 import Template
import numpy as np
from pathlib import Path


def gpu_contact_detection_init(**args):
    with open(Path(__file__).parent / "gpu_fine_contact_detection.cu") as f:
        template = Template(f.read())
    gpu_code = template.render(**args)
    mod = SourceModule(gpu_code)
    return mod.get_function("contact_detection")


def gpu_contact_detection(calc_function, particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,aa,bb,force):

    print(particle_A_center_of_mass.shape)
    # P, ...

    P = np.asarray(particle_A_center_of_mass.shape[0], dtype=np.int32)
    block = (64, 1, 1)
    grid = (1024, int(P / 1024 / 64 + 1))

    calc_function(
    cuda.In(particle_A_center_of_mass),
    cuda.In(particle_B_center_of_mass),
    cuda.In(particle_A_vertices),
    cuda.In(particle_B_vertices),
    cuda.In(particle_A_velocity),
    cuda.In(particle_B_velocity),
    cuda.In(aa),
    cuda.In(bb),
    cuda.InOut(force),
    cuda.In(P),
    block=block,
    grid=grid
    )
    print('return')
    return force


def Contact_detection_GPU(particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,aa,bb,Ks,Kd,mu,force):

    calc_func = gpu_contact_detection_init(Ks=Ks,Kd=Kd,mu=mu)
    force = gpu_contact_detection(calc_func, particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,aa,bb,force)

    sys.exit(0)


    for i in range(0,len(particle_A_center_of_mass)):




        Fn_A=[0,0,0]
        Fn_As=[0,0,0]
        Fn_Ad=[0,0,0]
        Torque_A=[0,0,0]
        Torque_B=[0,0,0]

        #force[i]=Fn_A
        force[i]=[Fn_A,Fn_As,Fn_Ad,Torque_A,Torque_B]
        #force[i]=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]


        A=particle_A_vertices[i]
        B=particle_B_vertices[i]


        #is there a collision or not
        coll=-1

        #Center of mass
        CmA=particle_A_center_of_mass[i]
        CmB=particle_B_center_of_mass[i]

        #1. Single point
        #print("Single point")
        d1=CmB-CmA
        (S1,aa1,bb1)=Support_function(A,B,d1)

        #check if the new point crosses the origin or not
        if dotp(d1,S1)<0:
            #ez nagy kerdes, hogy benne maradhat e:
            #print(str(aa)+" "+str(bb)+" NO COLLISION Point SAT\n   ")
            S2=[0,0,0]
            aa2=0
            bb2=0
            S3=[0,0,0]
            aa3=0
            bb3=0
            S4=[0,0,0]
            aa4=0
            bb4=0
            coll=0
            return(force)
            #check if the new point is the origin

        #2. Line segment
        #print("Line segment")
        d2=-1*d1
        (S2,aa2,bb2)=Support_function(A,B,d2)
        #check if the new point crosses the origin or not
        if dotp(d2,S2)<0:
            #print(str(aa)+" "+str(bb)+" NO COLLISION Line SAT\n    ")
            S3=[0,0,0]
            aa3=0
            bb3=0
            S4=[0,0,0]
            aa4=0
            bb4=0
            coll=0
            return(force)

        (coll,d3)=Check_linesegment(S1,S2)
        if coll==0:
            #print(str(aa)+" "+str(bb)+" NO COLLISION Line Voronoi\n    ")
            S3=[0,0,0]
            aa3=0
            bb3=0
            S4=[0,0,0]
            aa4=0
            bb4=0
            coll=0
            return(force)

        #3. Triangle
        (S3,aa3,bb3)=Support_function(A,B,d3)
        #check if the new point crosses the origin or not
        if dotp(d3,S3)<0:
            #print(str(aa)+" "+str(bb)+" NO COLLISION Triangle SAT\n")
            S4=[0,0,0]
            aa4=0
            bb4=0
            coll=0
            return(force)
        #check if the new point is the origin

        (coll,d4)=Check_triangle(S1,S2,S3)
        if coll==0:
            #print(str(aa)+" "+str(bb)+" NO COLLISION Triangele Voronoi\n   ")
            S4=[0,0,0]
            coll=0
            return(force)

        #4. Tetrahedron
        zz=0
        while (zz<100):
            #print("Iter Tetrahedron ",zz,"\n")
            (S4,aa4,bb4)=Support_function(A,B,d4)

            #check if the new point crosses the origin or not
            if dotp(d4,S4)<0:
                #print(str(aa)+" "+str(bb)+" NO COLLISION Tetra SAT\n")
                coll=0
                return(force)
            #check if the new point is the origin
            if S4[0]==0 and S4[1]==0 and S4[2]==0:
                #print(str(aa)+" "+str(bb)+" COLLISION Tetra Hitting origin\n   ")
                coll=1
                change=CorrectFlat(S1,S2,S3,S4)
                if change==1:
                    d4=np.cross(S1,S2)
                    (S4,aa4,bb4)=Support_function(A,B,d4)
                (penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,A,B)

                (Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Force(particle_A_center_of_mass[i],particle_B_center_of_mass[i],particle_A_velocity[i],particle_B_velocity[i],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
                force[i][0]=Fn_A
                force[i][1]=Fn_As
                force[i][2]=Fn_Ad
                force[i][3]=np.array(Torque_A)
                force[i][4]=Torque_B
                return(force)
            (coll,d4,todel)=Check_Tetrahedron(S1,S2,S3,S4)
            if coll==0:
                #print(str(aa)+" "+str(bb)+" NO COLLISION Tetra Voronoi\n   ")
                coll=0
                return(force)
            if coll==1:
                #print(str(aa)+" "+str(bb)+" COLLISION Tetra Vornoi\n   ")
                coll=1
                change=CorrectFlat(S1,S2,S3,S4)
                if change==1:
                    d4=np.cross(S1,S2)
                    (S4,aa4,bb4)=Support_function(A,B,d4)
                (penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,A,B)
                (Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Force(particle_A_center_of_mass[i],particle_B_center_of_mass[i],particle_A_velocity[i],particle_B_velocity[i],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
                force[i]=[Fn_A,Fn_As,Fn_Ad,Torque_A,Torque_B]
                return(force)
            if todel==1:
                S1uj=S2
                aa1uj=aa2
                bb1uj=bb2
                S2uj=S3
                aa2uj=aa3
                bb2uj=bb3
                S3uj=S4
                aa3uj=aa4
                bb3uj=bb4
            if todel==2:
                S1uj=S3
                aa1uj=aa3
                bb1uj=bb3
                S2uj=S1
                aa2uj=aa1
                bb2uj=bb1
                S3uj=S4
                aa3uj=aa4
                bb3uj=bb4
            if todel==3:
                S1uj=S1
                aa1uj=aa1
                bb1uj=bb1
                S2uj=S2
                aa2uj=aa2
                bb2uj=bb2
                S3uj=S4
                aa3uj=aa4
                bb3uj=bb4
            if todel==4:
                S1uj=S1
                aa1uj=aa1
                bb1uj=bb1
                S2uj=S2
                aa2uj=aa2
                bb2uj=bb2
                S3uj=S3
                aa3uj=aa3
                bb3uj=bb3
            #del S1,S2,S3, aa1, aa2, aa3, bb1, bb2, bb3
            S1=S1uj
            aa1=aa1uj
            bb1=bb1uj
            S2=S2uj
            aa2=aa2uj
            bb2=bb2uj
            S3=S3uj
            aa3=aa3uj
            bb3=bb3uj
            zz=zz+1
        return(force)
