from compute_energy import pes
from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import torch
import numpy as np
from itertools import combinations
import getenergy

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': True, 'morse_alpha': 1.3}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 0.5}

X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

def pes(geom_vectors, cartesian=True):
    g = np.asarray(geom_vectors)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = nn.transform_new_X(g, params, Xscaler)
    x = torch.tensor(data=newX)
    with torch.no_grad():
        E = model(x.float())
    e = nn.inverse_transform_new_y(E, yscaler)
    #e = e - (insert min energy here)
    #e *= 219474.63  ( convert units )
    return e

def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1,3)
    n = len(vec)
    distance_matrix = np.zeros((n,n))
    for i,j in combinations(range(len(vec)),2):
        R = np.linalg.norm(vec[i]-vec[j])
        distance_matrix[j,i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix),-1)]
    return distance_vector

def do1(xa,ya,za,xb,yb,zb,xc,yc,zc):
    x1 = xa + 0.01
    coo1 = [x1,ya,za,xb,yb,zb,xc,yc,zc]
    e1 = float(pes(geom_vectors=coo1,cartesian=True))
    x2 = xa - 0.01
    coo2 = [x2,ya,za,xb,yb,zb,xc,yc,zc]
    e2 = float(pes(geom_vectors=coo2,cartesian=True))
    differencex = (e2 - e1)/0.02
    y1 = ya + 0.01
    coo1 = [xa, y1, za, xb, yb, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    y2 = ya - 0.01
    coo2 = [xa, y2, za, xb, yb, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    differencey = (e2 - e1) / 0.02
    z1 = za + 0.01
    coo1 = [xa, ya, z1, xb, yb, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    z2 = za - 0.01
    coo2 = [xa, ya, z2, xb, yb, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    differencez = (e2 - e1) / 0.02
    return(differencex,differencey,differencez)

def do2(xa,ya,za,xb,yb,zb,xc,yc,zc):
    x1 = xb + 0.01
    coo1 = [xa,ya,za,x1,yb,zb,xc,yc,zc]
    e1 = float(pes(geom_vectors=coo1,cartesian=True))
    x2 = xb - 0.01
    coo2 = [x2,ya,za,x2,yb,zb,xc,yc,zc]
    e2 = float(pes(geom_vectors=coo2,cartesian=True))
    differencex = (e2 - e1)/0.02
    y1 = yb + 0.01
    coo1 = [xa, y1, za, xb, y1, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    y2 = yb - 0.01
    coo2 = [xa, y2, za, xb, y2, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    differencey = (e2 - e1) / 0.02
    z1 = zb + 0.01
    coo1 = [xa, ya, za, xb, yb, z1, xc, yc, z1]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    z2 = zb - 0.01
    coo2 = [xa, ya, za, xb, yb, z2, xc, yc, z2]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    differencez = (e2 - e1) / 0.02
    return(differencex,differencey,differencez)

def dh1(xa,ya,za,xb,yb,zb,xc,yc,zc):
    x1 = xb + 0.01
    coo1 = [xa,ya,za,x1,yb,zb,x1,yc,zc]
    e1 = float(pes(geom_vectors=coo1,cartesian=True))
    x2 = xc - 0.01
    coo2 = [xa,ya,za,x2,yb,zb,x2,yc,zc]
    e2 = float(pes(geom_vectors=coo2,cartesian=True))
    differencex = (e2 - e1)/0.02
    y1 = yc + 0.01
    coo1 = [xa, ya, za, xb, yb, zb, xc, y1, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    y2 = yc - 0.01
    coo2 = [xa, ya, za, xb, yb, zb, xc, y2, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    differencey = (e2 - e1) / 0.02
    z1 = zc + 0.01
    coo1 = [xa, ya, za, xb, yb, zb, xc, yc, z1]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    z2 = za - 0.01
    coo2 = [xa, ya, za, xb, yb, zb, xc, yc, z2]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    differencez = (e2 - e1) / 0.02
    return(differencex,differencey,differencez)

def cooO1(xa,ya,za,xb,yb,zb,xc,yc,zc):
    x1 = xa + 0.01
    coox1 = [x1, ya, za, xb, yb, zb, xc, yc, zc]
    x2 = xa - 0.01
    coox2 = [x2, ya, za, xb, yb, zb, xc, yc, zc]
    y1 = ya + 0.01
    cooy1 = [xa, y1, za, xb, yb, zb, xc, yc, zc]
    y2 = ya - 0.01
    cooy2 = [xa, y2, za, xb, yb, zb, xc, yc, zc]
    z1 = za + 0.01
    cooz1 = [xa, ya, z1, xb, yb, zb, xc, yc, zc]
    z2 = za - 0.01
    cooz2 = [xa, ya, z2, xb, yb, zb, xc, yc, zc]
    return(coox1, coox2,cooy1,cooy2,cooz1,cooz2)

def cooO2(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xb + 0.01
    coox1 = [xa, ya, za, x1, yb, zb, xc, yc, zc]
    x2 = xb - 0.01
    coox2 = [xa, ya, za, x2, yb, zb, xc, yc, zc]
    y1 = yb + 0.01
    cooy1 = [xa, ya, za, xb, y1, zb, xc, yc, zc]
    y2 = yb - 0.01
    cooy2 = [xa, ya, za, xb, y2, zb, xc, yc, zc]
    z1 = zb + 0.01
    cooz1 = [xa, ya, za, xb, yb, z1, xc, yc, zc]
    z2 = zb - 0.01
    cooz2 = [xa, ya, za, xb, yb, z2, xc, yc, zc]
    return (coox1, coox2,cooy1,cooy2,cooz1,cooz2)

def cooH(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xc + 0.01
    coox1 = [xa, ya, za, xb, yb, zb, x1, yc, zc]
    x2 = xc - 0.01
    coox2 = [xa, ya, za, xb, yb, zb, x2, yc, zc]
    y1 = yc + 0.01
    cooy1 = [xa, ya, za, xb, yb, zb, xc, y1, zc]
    y2 = yc - 0.01
    cooy2 = [xa, ya, za, xb, yb, zb, xc, y2, zc]
    z1 = zc + 0.01
    cooz1 = [xa, ya, za, xb, yb, zb, xc, yc, z1]
    z2 = za - 0.01
    cooz2 = [xa, ya, za, xb, yb, zb, xc, yc, z2]
    return (coox1, coox2,cooy1,cooy2,cooz1,cooz2)

if __name__ == "__main__":
    print("Start the calculation of energy...")
    # Based on the data in PES_data_new 17
    '''
    input_value = [0, 0, 1.1125, 0, 0.85, -0.12, 0, 0, 0]
    print(type(input_value))
    print(float(pes(geom_vectors=input_value, cartesian=True)))
    '''
    f_w = open('energyresultnew.txt', "w")
    m_w = open('fittingfigurenew.txt',"w")
    g_w = open('gradientresult.txt',"w")
    r_w = open('reread.txt',"w")
    final_w = open('trans.txt',"w")

    # for i in range(len(dirs)):
    f = open('read1.txt', 'r')
    lines = f.readlines()

    for index in range(len(lines)):
        m = lines[index].split()
        oo1 = float(m[0])
        oh1 = float(m[1])
        angle = float(m[2])
        #ww = (oo1*oo1+oh1*oh1-oh2*oh2)/(2*oo1*oh1)
        #angle1 = np.arccos(ww)
        #angle = angle1 * 180 / np.pi
        print(oo1, oh1, angle)

        # print cartisian coordinates
        a1 = (0, 0, 0)
        a2 = (oo1, 0, 0)
        degree = angle * np.pi / 180
        x = oh1 * np.cos(degree)
        y = oh1 * np.sin(degree)
        a3 = (x, y, 0)
        #   f_w.write(str(a1,a2,a3))
        input_value = [0, 0, 0, oo1, 0, 0, x, y, 0]
        l = float(pes(geom_vectors=input_value,cartesian=True))
        #print(l)
        final_w.write('energy'+'\n')
        final_w.write(str(l)+'\n')
        final_w.write('gradient'+'\n')
        x1 = do1(0,0,0,oo1,0,0,x,y,0)[0]
        y1 = do1(0,0,0,oo1,0,0,x,y,0)[1]
        z1 = do1(0,0,0,oo1,0,0,x,y,0)[2]
        x2 = do2(0, 0, 0, oo1, 0, 0, x, y, 0)[0]
        y2 = do2(0, 0, 0, oo1, 0, 0, x, y, 0)[1]
        z2 = do2(0, 0, 0, oo1, 0, 0, x, y, 0)[2]
        x3 = dh1(0, 0, 0, oo1, 0, 0, x, y, 0)[0]
        y3 = dh1(0, 0, 0, oo1, 0, 0, x, y, 0)[1]
        z3 = dh1(0, 0, 0, oo1, 0, 0, x, y, 0)[2]
        final_w.write('   1'+'    '+'8'+'   '+str(x1)+' '+str(y1)+' '+str(z1)+'\n')
        final_w.write('   2'+'    '+'8'+' '+str(x2) + ' ' + str(y2) + ' ' + str(z2) + '\n')
        final_w.write('   3'+'    '+'1'+' '+str(x3) + ' ' + str(y3) + ' ' + str(z3) + '\n')

    final_w.close()
    print(getenergy.read_eg("trans.txt"))
