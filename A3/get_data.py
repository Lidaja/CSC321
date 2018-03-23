from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from PIL import Image
import matplotlib.image as mpimg
import os
import random
from scipy.ndimage import filters
import urllib, cStringIO
from hashlib import sha256

#act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))
act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
first = False

random.seed(5)
images = []

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

#Function to flatten crop, resize and save an image in a given location
def handler(url, path, coords, hash):
    file = cStringIO.StringIO(urllib.urlopen(url).read())
    data = file.read()
    if hash == sha256(data).hexdigest():
        imshow(image)
        show()
        cropped_image = image[coords[1]:coords[3], coords[0]:coords[2]]
        imsave(path,imresize(cropped_image,(32,32,3)))

def handler2(url, filename, coords):
    file = urllib.urlopen(url)
    image = imread(file)
    imsave("example_pictures/colour_"+filename,image)
    handler(url,"example_pictures/"+filename,coords)
    


if not os.path.exists("training"):
    os.makedirs("training")
    
if not os.path.exists("validating"):
    os.makedirs("validating")
    
if not os.path.exists("testing"):
    os.makedirs("testing")
    
if not os.path.exists("testing_male"):
    os.makedirs("testing_male")
    
if not os.path.exists("testing_female"):
    os.makedirs("testing_female")
    
if not os.path.exists("validating_male"):
    os.makedirs("validating_male")
    
if not os.path.exists("validating_female"):
    os.makedirs("validating_female")

if not os.path.exists("example_pictures"):
    os.makedirs("example_pictures")

#Get training, testing, and validating sets
for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("faces_subset.txt"):
        if a in line:
            if i < 100:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(handler, (line.split()[4], "training/"+filename, [int(n) for n in line.split()[5].split(",")], line.split()[6]), {}, 30)
                if not os.path.isfile("training/"+filename):
                    continue
            elif i < 110:
                filename = name+str(i-100)+'.'+line.split()[4].split('.')[-1]
                timeout(handler, (line.split()[4], "validating/"+filename, [int(n) for n in line.split()[5].split(",")], line.split()[6]), {}, 30)
                if not os.path.isfile("validating/"+filename):
                    continue
            elif i < 120:
                filename = name+str(i-110)+'.'+line.split()[4].split('.')[-1]
                timeout(handler, (line.split()[4], "testing/"+filename, [int(n) for n in line.split()[5].split(",")], line.split()[6]), {}, 30)
                if not os.path.isfile("testing/"+filename):
                    continue
            else:
                continue
            print filename
            i += 1

'''others = {}
#Get other male and female subjects
for line in open("subset_actors.txt"):
    full_name = line.split()[0]+" "+line.split()[1]
    name = line.split()[len(line.split())%7+1].lower()
    if ((not name in others) or others[name] < 10) and not full_name in act:
            if not name in others:
                others[name] = 0
            filename = name+str(others[name])+'.'+line.split()[len(line.split())%7+4].split('.')[-1]
            timeout(handler, (line.split()[len(line.split())%7+4], "testing_male/"+filename, [int(n) for n in line.split()[len(line.split())%7+5].split(",")]), {}, 5)
            if not os.path.isfile("testing_male/"+filename):
                continue
            others[name] += 1
    elif ((not name in others) or others[name] < 20) and not full_name in act:
            if not name in others:
                others[name] = 0
            filename = name+str(others[name]-10)+'.'+line.split()[len(line.split())%7+4].split('.')[-1]
            timeout(handler, (line.split()[len(line.split())%7+4], "validating_male/"+filename, [int(n) for n in line.split()[len(line.split())%7+5].split(",")]), {}, 5)
            if not os.path.isfile("validating_male/"+filename):
                continue
            others[name] += 1

for line in open("subset_actresses.txt"):
    full_name = line.split()[0]+" "+line.split()[1]
    name = line.split()[len(line.split())%7+1].lower()
    if ((not name in others) or others[name] < 10) and not full_name in act:
            if not name in others:
                others[name] = 0
            filename = name+str(others[name])+'.'+line.split()[len(line.split())%7+4].split('.')[-1]
            timeout(handler, (line.split()[len(line.split())%7+4], "testing_female/"+filename, [int(n) for n in line.split()[len(line.split())%7+5].split(",")]), {}, 5)
            if not os.path.isfile("testing_female/"+filename):
                continue
            others[name] += 1
    elif ((not name in others) or others[name] < 20) and not full_name in act:
            if not name in others:
                others[name] = 0
            filename = name+str(others[name]-10)+'.'+line.split()[len(line.split())%7+4].split('.')[-1]
            timeout(handler, (line.split()[len(line.split())%7+4], "validating_female/"+filename, [int(n) for n in line.split()[len(line.split())%7+5].split(",")]), {}, 5)
            if not os.path.isfile("validating_female/"+filename):
                continue
            others[name] += 1'''