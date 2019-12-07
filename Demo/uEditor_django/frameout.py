import os   
import cv2 

def frame(filepath,destpath,f1,f2,g1,g2,s1,s2,h1,w1,h2,w2):
    dest=destpath
    image = cv2.imread(filepath)

    #flower
    if (f1 !=0 and f2 !=0):
        img=cv2.rectangle(image, (f1,f2), (h1,h2), (0,255,0), 4)
        font=cv2.FONT_HERSHEY_SIMPLEX
        text = 'Flower'
        cv2.putText(img, text, (f1, f2), font, 2, (0,255,0), 5)

    #gravel
    if (g1 !=0 and g2 !=0):
        if (f1==0 and f2 ==0):
            img=cv2.rectangle(image, (g1,g2), (h1,w1), (0,255,0), 4)
        else:
            img=cv2.rectangle(img, (g1,g2), (1100,800), (0,255,0), 4)
        font=cv2.FONT_HERSHEY_SIMPLEX
        text = 'Greval'
        cv2.putText(img, text, (g1, g2), font, 2, (0,255,0), 5)  

    #Sugar
    if (s1 !=0 and s2 !=0):
        if (g1==0 and g2==0):
            img=cv2.rectangle(image, (s1,s2), (h2,w2), (0,255,0), 4)
        else:
            img=cv2.rectangle(img, (s1,s2), (h2,w2), (0,255,0), 4)
        font=cv2.FONT_HERSHEY_SIMPLEX
        text = 'Sugar'
        cv2.putText(img, text, (s1+100, s2+100), font, 2, (0,255,0), 5)

    cv2.imwrite(dest+'.jpg',img) 


# if __name__== "__main__" :
def getimage(imagename):
    
    imagename=input("Enter a name of a cloud image.")
    path=os.path.dirname(os.path.abspath(__file__)) 
    filepath =path+'/'+imagename     #original image
    destpath= path+'/res'    # resized images saved here

    if (imagename=='002f507.jpg'):
        frame(filepath,destpath,0,0,258,332,113535,3,1100,800,1000,150)
    elif (imagename == '0035ae9.jpg'):
        frame(filepath,destpath,0,0,444,154,603,2,1100,800,1000,150)
    elif (imagename == '0038327.jpg'):
        frame(filepath,destpath,0,0,0,0,985,39,1100,800,10,1100)
    elif (imagename == '004f759.jpg'):
        frame(filepath,destpath,72280,320,0,0,355,150,1500,20,10,1100)
    elif (imagename == '005ba08.jpg'):
        frame(filepath,destpath,0,0,55306,3,0,0,1100,800,1000,150)



# 002f507.jpg 
# fish
# flower
# gravel 358 332
# Sugar 113525 3

# 0035ae9.jpg
# fish
# flower
# Gravel 444 154
# Sugar 603 2

# 0038327.jpg
# Fish
# Flower
# Gravel
# Sugar 985 39


#004f759.jpg
# Fish,
# Flower 72280 20
# Gravel 
# Sugar,355 150

# 005ba08.jpg
# Fish
# Flower
# gravel 55306 3
# Sugar 
