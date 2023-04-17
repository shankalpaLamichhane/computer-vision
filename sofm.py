
from random import *
from math import *
from PIL import Image
from time import *

class Node:

    def __init__(self, X=0, Y=0):
        
        self.R=randint(0,255)  # R
        self.G=randint(0,255)  # G
        self.B=randint(0,255)  # B
        self.X=X               # X
        self.Y=Y               # Y

class SOM:

    def __init__(self, height=500, width=500):
        
        self.height=height                      
        self.width=width                       
        self.radius=height/2+width/2            
        self.total=self.height*self.width       
        
        self.nodes=[[0 for x in range(self.height)] for y in range(self.width)] 

        for i in range(self.width):
            for j in range(self.height):
                self.nodes[i][j]=Node(i,j)
    
    def W_distance(self, r_sample, node):
        return sqrt((r_sample[0]-node.R)**2+(r_sample[1]-node.G)**2+(r_sample[2]-node.B)**2)
    
    def distance(self, node1, node2):
        return sqrt((node1.X-node2.X)**2+(node1.Y-node2.Y)**2)

    def get_bmu(self,r_sample):

        match_amt=0
        max_dist=1000000.0
        match_list=[]
        for i in range(self.width):
            for j in range(self.height):
                t_dist=self.W_distance(r_sample,self.nodes[i][j])
                if t_dist<max_dist:
                    max_dist=t_dist
                    match_list=[]
                    match_list.append(self.nodes[i][j])
                    match_amt=1
                elif t_dist==max_dist:
                    match_list.append(self.nodes[i][j])
                    match_amt+=1
    
        return match_list[randint(0,match_amt-1)]
    
    def scale_neighbors(self, bmu_loc, r_sample, times):

        R2=(int)(((float)(self.radius)*(1.0-times)/2.0))+1
        outer=Node(R2,R2)
        center=Node(0,0)
        d_normalize=self.distance(center,outer)
        for i in range(-R2,R2+1):
            for j in range(-R2,R2+1):
                if j+bmu_loc.Y >= 0 and j+bmu_loc.Y < self.height and i+bmu_loc.X >= 0 and i+bmu_loc.X < self.width:
                    outer=Node(i,j)
                    distance=self.distance(center,outer)
                    distance/= d_normalize
                    t=(float)(exp(-1.0*distance**2/0.15))
                    t/=(times*4.0+1.0)
                    self.nodes[i+bmu_loc.X][j+bmu_loc.Y].R = int(self.nodes[i+bmu_loc.X][j+bmu_loc.Y].R*(1-t) + r_sample[0]*t)
                    self.nodes[i+bmu_loc.X][j+bmu_loc.Y].G = int(self.nodes[i+bmu_loc.X][j+bmu_loc.Y].G*(1-t) + r_sample[1]*t)
                    self.nodes[i+bmu_loc.X][j+bmu_loc.Y].B = int(self.nodes[i+bmu_loc.X][j+bmu_loc.Y].B*(1-t) + r_sample[2]*t)

        # self.save_image(str((int)(100*times)))
        self.save_image("result_2")
        sleep(1)
    def save_image(self, imagename):
        newImage = Image.new ("RGB", (self.width,self.height), (0,0,0))
        newpix = newImage.load()
        for i in range(self.width):
            for j in range(self.height):
                newpix[i,j] = (self.nodes[i][j].R,self.nodes[i][j].G,self.nodes[i][j].B,255)
        newImage.save("output.jpg")

    
if __name__ == "__main__":
    

    
    myimage = Image.open('LENNA.bmp')
    myimage_with = myimage.size[0]
    myimage_height = myimage.size[1]
    mypix = myimage.load()
    mySom = SOM(myimage_with,myimage_height)
    times=0.0
    MAX_ITER=1000
    T_INC=1.0/(float)(MAX_ITER)
    while(True):
        
        if times<1.0:

            random_x=randint(0,myimage_with-1)
            random_y=randint(0,myimage_height-1)
            r_sample=mypix[random_x,random_y]
            bmu_loc=mySom.get_bmu(r_sample)
            mySom.scale_neighbors(bmu_loc,r_sample,times)
            times+=T_INC
            
        else:
            break
