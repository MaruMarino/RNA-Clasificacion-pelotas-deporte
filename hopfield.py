import numpy as np
import random
from PIL import Image
from PIL import ImageDraw
import os
import re

#convert matrix to a vector
def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1


# Calculos los pesos para una imagen
def calculaW(x):
    if len(x.shape) != 1:
        print("Error: no es un vector!")
        return None
    else:           
      # define la transpuesta y ajusta la forma del vector
      tx = np.matrix(x).T
      x = x.reshape((1,x.shape[0]))
      # multiplica la transpuesta por sigo misma
      w = np.dot(tx, np.array(x)) 
      # llena la diagonal con ceros
      for i in range(w.shape[0]):
          w[i,i] = 0
    return w

#Create Weight matrix for a single image
def create_W(x):
    if len(x.shape) != 1:
        print ("The input is not vector")
        return
    else:
        w = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(i,len(x)):
                if i == j:
                    w[i,j] = 0
                else:
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w


#Read Image file and convert it to Numpy array
def readImg2array(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN= pilIN.resize(size)
    #pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    x = np.zeros(imgArray.shape,dtype=np.float)
    x[imgArray > threshold] = 1
    x[x==0] = -1
    return x

#Convert Numpy array to Image file like Jpeg
def array2img(data, outFile = None):

    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img


#Update
def update(w,y_vec,theta=0.5,time=100):
    for s in range(time):
        m = len(y_vec)
        i = random.randint(0,m-1)
        u = np.dot(w[i][:],y_vec) - theta

        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1
    return y_vec

def reconocerImagenes(arImages, arClases=None):
  print("\n> Reconociendo imágenes: ")
  for i in range(len(arImages)):    
    # prepara la imagen
    t = readImg2array(arImages[i])
    auxShape = t.shape
    t_vec = mat2vec(t)
    # procesa por la red
    t_vec_after = reconocer(w=matW, vec=t_vec)
    t_vec_after = t_vec_after.reshape(auxShape)
    # muestra los resultados
    if arClases is not None and i<len(arClases):
      strTitulo = " " + str(arClases[i]) + ": "
    else:
      strTitulo = ""
    fig = plt.figure()
    plt.gray()      
    fig.suptitle( strTitulo )
    # muestra la entrada
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Entrada')
    ax1.imshow( arImages[i] )
    plt.axis("off")  
    # muestra la salida
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Salida')
    ax2.imshow( array2img(t_vec_after) )
    plt.axis("off")  
    plt.tight_layout()
    fig = plt.gcf()

#The following is training pipeline
#Initial setting
def hopfield(train_files, test_files,theta=0.5, time=1000, size=(100,100),threshold=60, current_path=None):

    #read image and convert it to Numpy array
    print ("Importing images and creating weight matrix....")

    #num_files is the number of files
    num_files = 0
    for path in train_files:
        print (path)

        # prepara la imagen     
        x = readImg2array(file=path,size=size,threshold=threshold)
        x_vec = mat2vec(x)

        # calcula matriz de pesos de la imagen
        tmp_w = calculaW(x_vec)
        print(tmp_w)

        print (len(x_vec))
        if num_files == 0:
            w = create_W(x_vec)
            num_files = 1
        else:
            tmp_w = create_W(x_vec)
            w = w + tmp_w
            num_files +=1

    print("\nEntrenamiento finalizado para ", w.shape[0]," neuronas.")
    print("\n  Matriz de Pesos General de forma ", w.shape, ":")
    print(w)


    print("\n> Reconociendo imágenes: ")

    #Import test data
    counter = 0
    for path in test_files:

        y = readImg2array(file=path,size=size,threshold=threshold)
        oshape = y.shape
        y_img = array2img(y)
        #y_img.show()
        print ("Imported test data")

        y_vec = mat2vec(y)
        print ("Updating...")
        y_vec_after = update(w=w,y_vec=y_vec,theta=theta,time=time)
        print(y_vec_after)
        y_vec_after = y_vec_after.reshape(oshape)
        print(y_vec_after)
        if current_path is not None:
            outfile = current_path+"/after_"+str(counter)+".jpeg"
            array2img(y_vec_after,outFile=outfile)
        else:
            after_img = array2img(y_vec_after,outFile=None)
            #after_img.show()
        counter +=1


#Main
#First, you can create a list of input file path
current_path = os.getcwd()
train_paths = []
path = current_path+"/train/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-]*.jp[e]*g',i):
        train_paths.append(path+i)

#Second, you can create a list of sungallses file path
test_paths = []
path = current_path+"/test/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
        test_paths.append(path+i)

#Hopfield network starts!
hopfield(train_files=train_paths, test_files=test_paths, theta=0.1,time=20000,size=(100,100),threshold=60, current_path = current_path)