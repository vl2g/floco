import easyocr
import cv2
import os
import json


def getTextCoordinates(image, reader):
    results = reader.readtext(image, paragraph = True)
    l = []
    
    for (bbox, text) in results:
        print(text, bbox)
        l.append((text, bbox))
    return l


def annotate_text(pathsrc):
    dictionary = {}
    reader = easyocr.Reader(['en'])
    
    for filename in os.listdir(pathsrc):
        path = os.path.join(pathsrc, filename)
        image = cv2.imread(path)
        textlist = getTextCoordinates(image, reader)
        dictionary[filename] = textlist
        
    return dictionary


def getShapeCoordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = {'Rectangle': [], 'Oval': [], 'Diamond': [], 'Parallelogram': []}
    i = 0
    for contour in contours:
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
    
        if area > 1000:
            
            if len(approx) == 4:
                x, y = [], [] # coordinates of blocks
                
                for l in approx:
                    pair = l[0]
                    x.append(pair[0])
                    y.append(pair[1])
                
                if (abs(x[1]-x[0])<6 and abs(x[3]-x[2])<6 and abs(y[3]-y[0])<6 and abs(y[2]-y[1])<6) or (abs(x[3]-x[0])<6 and abs(x[1]-x[2])<6 and abs(y[1]-y[0])<6 and abs(y[2]-y[3])<6):
                    shapes['Rectangle'].append(approx)

                elif (abs(x[1]-x[3])>6 and abs(x[2]-x[1])<50) or (abs(x[2]-x[0])>6 and abs(x[0]-x[1])<50):
                    shapes['Parallelogram'].append(approx)

                elif (abs(x[1]-x[3])<6 and abs(x[2]-x[1])>50) or (abs(x[2]-x[0])<6 and abs(x[0]-x[1])>50):
                    shapes['Diamond'].append(approx)

            elif len(approx) >=8 and len(approx) <= 13:
                shapes['Oval'].append(approx)
                
    return shapes


def annotate_shapes(pathsrc):
    dictionary = {}
    
    for filename in os.listdir(pathsrc):
        path = os.path.join(pathsrc, filename)
        image = cv2.imread(path)
        shapes = getShapeCoordinates(image)
        dictionary[filename] = shapes
        
    return dictionary


def find_min_max(coordinate_list):
    min_x, min_y, max_x, max_y = coordinate_list[0][0], coordinate_list[0][1], coordinate_list[0][0], coordinate_list[0][1]
    for x, y in coordinate_list:
        if min_x > x:
            min_x = x
        if min_y > y:
            min_y = y
        if max_x < x:
            max_x = x
        if max_y < y:
            max_y = y
    return (min_x, min_y, max_x, max_y)

def is_within(text_coordinates, min_x, min_y, max_x, max_y):
    tl, tr, bl, br = text_coordinates[0], text_coordinates[1], text_coordinates[2], text_coordinates[3]
    flag = False
    if (min_x < tl[0] < max_x) and (min_x < tr[0] < max_x) and (min_x < bl[0] < max_x) and (min_x < br[0] < max_x) and (min_y < tl[1] < max_y) and (min_y < tr[1] < max_y) and (min_y < bl[1] < max_y) and (min_y < br[1] < max_y):
        flag = True
    return flag

def narray_to_list(narray):
    l = []
    for points in narray:
        l.append([points[0][0], points[0][1]])
    return l

def associate_shape_each(text, text_coordinates, shape_coordinates):
    SHAPE = None
    SHAPELIST = ['Rectangle', 'Diamond', 'Parallelogram', 'Oval']
    dcount = 0
    
    for shape in SHAPELIST:
        for narray in shape_coordinates[shape]:
            l = narray_to_list(narray)
            min_x, min_y, max_x, max_y = find_min_max(l)
            flag = is_within(text_coordinates, min_x, min_y, max_x, max_y)
            if flag:
                SHAPE = shape.upper()
        
    return SHAPE

def get_diamond_coordinates(text_coordinates, shape_coordinates):
    
    for narray in shape_coordinates['Diamond']:
        l = narray_to_list(narray)
        min_x, min_y, max_x, max_y = find_min_max(l)
        flag = is_within(text_coordinates, min_x, min_y, max_x, max_y)
        if flag:
            return l
        
def find_centroid(l):
    sum_x, sum_y = 0, 0
    for x, y in l:
        sum_x += x
        sum_y += y
    
    centroid = (sum_x//4, sum_y//4)
    return centroid

def find_distance(p1, p2):
    dis = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return dis

def associate_nearest_diamond(text_coord, diamond_coordinates):
    text_centroid = find_centroid(text_coord)
    min_dis = float('inf')
    nearest = None
    for diamond in diamond_coordinates:
        diamond_centroid = find_centroid(diamond_coordinates[diamond])
        dis = find_distance(text_centroid, diamond_centroid)
        if dis < min_dis:
            min_dis = dis
            nearest = diamond
    return nearest
        
def associate_shape(name, text_dict, shape_dict):
    shape_coordinates = shape_dict[name]
    dcount = 0
    diamond_coordinates = {}
    text_shape_coord_list = []
    text_shape_list = []
    encoding = ''
    for text, text_coordinates in text_dict[name]:
        SHAPE = associate_shape_each(text, text_coordinates, shape_coordinates)
        if SHAPE == 'DIAMOND':
            dcount += 1
            SHAPE += str(dcount)
            diamond_coordinates[SHAPE] = get_diamond_coordinates(text_coordinates, shape_coordinates)
        text_shape_coord_list.append((text, SHAPE, text_coordinates))
        
        
    for text, shape, text_coord in text_shape_coord_list:
        if shape == None:
            SHAPE = associate_nearest_diamond(text_coord, diamond_coordinates)
            text_shape_list.append((text, SHAPE))
            if SHAPE == None:
                encoding += '{'+text+',None},'
            else:
                encoding += '{'+text+','+SHAPE+'},'
        else:
            text_shape_list.append((text, shape))
            encoding += '{'+text+','+shape+'},' 
    #print(text_shape_list)
    encoding = encoding[:-1]
    return (text_shape_list, encoding)
            

def annotate_encodings(pathsrc, text_dict, shape_dict, encodings_pth):
    dictionary_tuple = {}
    dictionary_string = {}
    dictionary_modified_string = {}

    for filename in os.listdir(pathsrc):
        encoding_tuple, encoding_string = associate_shape(filename, text_dict, shape_dict)
        dictionary_tuple[filename[:-4]] = encoding_tuple
        dictionary_string[filename[:-4]] = encoding_string
        dictionary_modified_string[filename[:-4]] = encoding_string[1:-1].replace('},{', ' [SEP] ')
        
    with open(encodings_pth, 'w') as convert_file:
        convert_file.write(json.dumps(dictionary_modified_string))


def get_encodings():
    # Set path to png flowchart images
    pngpath = ""
    # Set path to a file to save encodings
    encodings_pth = ""

    # Get text inside flowchart blocks and on arrowheads
    # along with their coordinates with respect to the flowchart image
    # using easyocr
    text_dict = annotate_text(pngpath)

    # Get shape coordinates of flowchart blocks and categorize them
    # using contour detection into Rectangle, Diamond, Parallelogram and Oval  
    shape_dict = annotate_shapes(pngpath)

    annotate_encodings(pngpath, text_dict, shape_dict, encodings_pth)


get_encodings()