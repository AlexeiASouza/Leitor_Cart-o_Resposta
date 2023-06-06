#IMPORTA as Libs Necessárias
#import os
#import random
from collections import deque
import tensorflow as tf

import math
import imutils
import cv2 
import numpy as np

from global_variables import *

import pytesseract #Lib do OCR
pytesseract.pytesseract.tesseract_cmd = tesseractOCR_path


#Função para detecção dos objetos desejados na imagem
def detect_fn(image,detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#Função para criação de um dicionário com as predições
def get_preds(detections,score):
    preds = {}
    count = 0
    
    scores = detections['detection_scores'].numpy()
    boxes = detections['detection_boxes'].numpy()
    labels = detections['detection_classes'].numpy()
    
    scores = np.argwhere(scores > score)
    N = len(scores)
    
    for scr in scores:
        data = {}
        data['label'] = labels[scr[0]][scr[1]]
        data['boxes'] = boxes[scr[0]][scr[1]]
        preds[count] = data
        count += 1
    return preds

#Função para obtenção do bbox de acordo com o label relacionado
def get_box_by_label(preds,label):
    for pred in preds:
        lb = preds[pred]['label']
        if lb == label:
            return preds[pred]['boxes']
        

#Função para obtanção do objeto recortado da imagem
def get_object(img,boxes,ofs):
    shape = img.shape[0:-1]
    
    xmin = int(boxes[0]*shape[0])
    ymin = int(boxes[1]*shape[1])
    xmax = int(boxes[2]*shape[0])
    ymax = int(boxes[3]*shape[1])
    
    return img[xmin-ofs:xmax+ofs,ymin-ofs:ymax+ofs]

#Função para aplicar operações morfológicas na imagem
def morphology_operations(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 128, 256)
    return edged

#Função para obtenção dos contornos da grade
def get_contours(img,area_):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    grades = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= area_:
            grades.append(cnt)
    
    return grades

#Função para obtenção do polígono convexo que compõe a grade
def get_rectangles(img,area):
    contours = get_contours(img,area)

    convex_polygons = []
    for contour in contours:
        # Aproxima o contorno para um polígono com menos vértices
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Verifica se o polígono é convexo
        if cv2.isContourConvex(approx):
            convex_polygons.append(approx)
    return convex_polygons


#Função para obtenção do centroide de um contorno
def get_centroid(contour):
    x = contour[0][0][0] + contour[1][0][0] + contour[2][0][0] + contour[3][0][0]
    y = contour[0][0][1] + contour[1][0][1] + contour[2][0][1] + contour[3][0][1]
    
    return x//4, y//4

#Função para obtenção da posição esquerda/direita de um ponto em relação a outro
def pos(p1,p2):
    if p2 > p1:
        return -1
    else:
        return 1

#Função para organizar os pontos de um polígono convexo de acordo com o mesmo sentido(horário)
def organize_points(contour):
    xc,yc = get_centroid(contour)
    new_contour = [[],[],[],[]]
    indexes = []
    
    for idx,point in enumerate(contour):
        x = point[0][0]
        y = point[0][1]
        
        xpos =  pos(x,xc)
        ypos =  pos(y,yc)
        
        if xpos == -1 and ypos == -1:
            indexes.append([idx,0])
        elif xpos == -1 and ypos == 1:
            indexes.append([idx,1])
        elif xpos == 1 and ypos == 1:
            indexes.append([idx,2])
        elif xpos == 1 and ypos == -1:
            indexes.append([idx,3])
    
    for index in indexes :
        new_contour[index[1]] = list(contour[index[0]])
    return np.array(new_contour)

#Função para obter o retângulo da grade
def get_result_rectangle(rectangles):
    for idx,rect in enumerate(rectangles):
        rectangles[idx] = organize_points(rect)


    rectg_result = np.zeros_like(rectangles[0])
    N = len(rectangles)

    for i in range(N):
        rectg_result += rectangles[i]
    rectg_result = rectg_result//N

    return rectg_result

#Função para obter o ângulo de um retângulo 
def get_theta(rectangle):
    xa = rectangle[0][0][0]
    xb = rectangle[3][0][0]
    ya = rectangle[0][0][1]
    yb = rectangle[3][0][1]
    
    
    m = (yb - ya)/(xb-xa)
    theta = np.arctan(m)
    theta = theta * (180/np.pi)
    return theta

#Função para obtenção da grade já ajustada pelo seu ângulo
def get_grade(img,rectangle):
    theta = get_theta(rectangle)
    rotated = imutils.rotate(img, angle=theta)

    xmin = (rectangle[0][0][0] + rectangle[1][0][0])//2
    xmax = (rectangle[2][0][0] + rectangle[3][0][0])//2
    ymin = (rectangle[0][0][1] + rectangle[3][0][1])//2
    ymax = (rectangle[1][0][1] + rectangle[2][0][1])//2

    grade = rotated[ymin:ymax,xmin:xmax]

    return grade,theta

#Função para obter os recortes das bolhas de resposta individuais 
def get_rois(region,vertices,l_mask,l_roi):
    rois = []
    
    for i in range(10):
        for j in range(4):
            vert = vertices[i][j]
            xc,yc = bubble_center(vert[1],vert[0],l_mask)
            x1,x2,y1,y2 = get_bubble_coords(xc,yc,l_roi)
            roi = region[x1:x2,y1:y2]

            
            roi = rois.append(roi)

        
    return rois

#Função para calcular o centro da bolha
def bubble_center(x,y,n):
    xc = x + n/2
    yc = y + n/2
    
    return xc,yc

#Função para calcular as coordenas da bolha
def get_bubble_coords(xc,yc,n):
    return int(xc-n/2), int(xc+n/2), int(yc-n/2), int(yc + n/2)

#Função para centralizar um roi da imagem de acordo com sua distribuição de pixels
def centralize(roi, N):
    
    result_roi = np.ones((x_bubble_space+2,x_bubble_space+2))*255

    h_roi,w_roi = roi.shape
    h_im,w_im = result_roi.shape

    x = int((w_im - w_roi) // 2)
    y = int((h_im - h_roi) // 2)

    result_roi[y:y+h_roi, x:x+w_roi] = roi
    result_roi = result_roi.astype('uint8')
    
    x,y = result_roi.shape
    
    for i in range(N):
        xc,yc = get_roi_center(result_roi)
        
        xofs = (x//2)-xc
        yofs = (y//2)-yc
        
        ones = np.ones_like(result_roi)*255
        x_sig = get_signal(xofs)
        y_sig = get_signal(yofs)

        
        #NE
        if x_sig == 1 and y_sig == -1:
            new_roi = result_roi[0:x-abs(xofs),abs(yofs):y]
        
        
        #NO
        elif x_sig == 1 and y_sig == 1:           
            new_roi = result_roi[0:x-abs(xofs),0:y-abs(yofs)]

        #SE
        elif x_sig == -1 and y_sig == -1:
            new_roi = result_roi[abs(xofs):x,abs(yofs):y]

        #SO
        elif x_sig == -1 and y_sig == 1:
            new_roi = result_roi[abs(xofs):x,0:y-abs(yofs)]

        xr,yr = new_roi.shape
        pi_x = int(np.ceil((x - xr) / 2))
        pi_y = int(np.ceil((y - yr) / 2))
        
        
        ones[pi_x:pi_x+xr,pi_y:pi_y+yr] = new_roi
        result_roi = ones
    
    return result_roi
        

#Função para centralizar um conjunto de rois da imagem
def centralize_rois(region,vertices,l_mask,l_roi):
    rois = get_rois(region,vertices,l_mask,l_roi)
    result_rois = []
    for roi in rois:
        try:
            roi = centralize(roi,5)
            result_rois.append(roi)
        except:
            result_rois.append(roi)
    
    

    return result_rois

#Função para obter o índice positivo ou negativo de um número n
def get_signal(n):
    if n >= 0:
        return 1
    
    else:
        return -1
    

#Função para calcular o centro de um fecho convexo
def get_roi_center(roi):
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[0:2]
    bubble_contour = contours[-1]
    hull = cv2.convexHull(bubble_contour)
    
    momento = cv2.moments(hull)

    # Calcule as coordenadas do centro do fecho convexo
    centro_x = int(momento['m10'] / momento['m00'])
    centro_y = int(momento['m01'] / momento['m00'])
    
    return centro_y,centro_x



#Função para obter os pontos contidos em um pixel de raio n que não estejam incluidos no vetor previous points(pp)
def pontos_dentro_do_circulo(raio,centro,pp):
    pontos = []
    centro_x, centro_y = centro
    
    for x in range(centro_x - raio, centro_x + raio + 1):
        for y in range(centro_y - raio, centro_y + raio + 1):
            distancia = math.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
            if distancia <= raio:
                if (x,y) not in pp:
                    pontos.append((x, y))
    
    return pontos

#Função de balde de água para segmentação de componentes conectados
def bucket_fill(image, start_x, start_y, target_color, replacement_color):
    # Verifica se a cor inicial é diferente da cor de substituição
    if target_color == replacement_color:
        return image

    # Obtém as dimensões da imagem
    rows = len(image)
    cols = len(image[0])

    # Define as direções possíveis (cima, baixo, esquerda, direita)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Cria uma fila para armazenar os pixels a serem verificados
    queue = deque()
    queue.append((start_x, start_y))

    # Preenche a região com a cor de substituição
    while queue:
        x, y = queue.popleft()

        # Verifica se o pixel está dentro dos limites da imagem
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue

        # Verifica se o pixel possui a cor alvo
        if image[x][y] == target_color:
            # Define a cor de substituição para o pixel
            image[x][y] = replacement_color

            # Adiciona os pixels vizinhos à fila
            for dx, dy in directions:
                queue.append((x + dx, y + dy))

    # Retorna a imagem preenchida
    return image

#Função para aplicação da função balde de água
def apply_bucket_fill(roi,points,value,color):
    result_roi = roi.copy()
    flag = False
    for point in points:
        target = result_roi[point[0],point[1]]
        if target == value:
            flag = True
            result_roi = bucket_fill(result_roi,point[0],point[1],target,color)
            color += 1
        
        elif target == (color - 1):
            continue
        
    return result_roi,color

#Função para verificar se os pontos de uma imagem correspondem a determinado valor
def get_points_with_value(roi,points,value):
    points_with_value = []
    
    for point in points:
        pvalue = roi[point[0]][point[1]]
        if pvalue == value:
            points_with_value.append(point)
    
    return points_with_value

#Função para criação de um dicionário com keys contidas entre o start/end correspondentes ao número de pontos correspondentes na imagem
def get_dict(start,end,img):
    colors = {}
    for i in range(start,end):
        colors[i] = (img== i).sum()
    
    return colors
    


#Função para obtenção do dicionário de pontos com os valores de cada componente segmentado
def get_segmentation_dict(roi,r,xc,yc,nvalue):
    result_roi = roi.copy()//255
    #Aplica a função balde de agua na borda
    borde_color = result_roi[0][0]
    result_roi = bucket_fill(result_roi,0,0,borde_color,2)
    
    previous_points = []
    
    color_count = 3
    
    for i in range(r):
        points = pontos_dentro_do_circulo(i+1,(xc,yc),previous_points)
        noise = get_points_with_value(result_roi,points,nvalue)
        result_roi,color_count = apply_bucket_fill(result_roi,noise,nvalue,color_count)
        
        previous_points += points
                            
    
    result_dict = get_dict(3,color_count,result_roi)
    return result_dict,result_roi

#Função para verificar se uma bola de resposta não está preenchida
def isempty(roi,xc,yc,r):
    #Dilata a imagem para garantir que as bordas estarão fechadas
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.erode(roi, kernel, 1)
    roi = centralize(roi,5)
    xc,yc = get_roi_center(roi)
    #------------------------------------------------------------
    seg_dict,seg_roi = get_segmentation_dict(roi,r,xc,yc,0)
    keys = list(seg_dict.keys())
    distances = 0
    
    dist_dict = {}
    min_dict = {}
    min = 100
    
    for key in keys:
        points = np.argwhere(seg_roi == key)
        for point in points:
            dist = np.sqrt(((xc - point[0])**2) + ((yc - point[1])**2))
            distances += dist
            if dist < min: #Caso a média das distâncias coincidir com o local possível da borda
                min = dist
        
        dist_dict[key] = distances/seg_dict[key]
        min_dict[key] = min
        distances = 0
        min = 100
    
    dist_keys = list(dist_dict.keys())
    
    for dkey in dist_keys:
        distance = dist_dict[dkey]
        if (distance <= r+3) and (distance >= r-3):
            if min_dict[dkey] <= r//2:
                return False
            else:
                continue
        else:
            return False
        
    
    return True

#Função para verificar se uma bola de resposta está preenchida
def isfull(roi,xc,yc,r):
    seg_dict,_ = get_segmentation_dict(roi,r,xc,yc,1)
    keys = list(seg_dict.keys())
    distances = 0
    
    total_noise = 0
    area = np.pi*(r**2)
    offset = int(area - (minpercent*area))

    
    for key in keys:
        n = seg_dict[key]
        total_noise += n
    
        
    if total_noise <= offset:
        return True
    
    else:
        return False
        

#Função para obter o valor do círculo (cheio:1, vazio:0, inválido:-1)
def get_circle_value(roi,xc,yc):
    r = (bubble_diameter//2) + 1
    try:
        if isempty(roi,xc,yc,r) == True:
            return 0
        elif isfull(roi,xc,yc,r) == True:
            return 1
    
        else:
            return -1
    except:
        return -1
    

#Função para obter os resultados de um conjunto de recortes(bolas de respostas) 
def get_results(rois):
    results = []
    for roi in rois:
        try:
            xc,yc = get_roi_center(roi)
            res = get_circle_value(roi,xc,yc)
            results.append(res)
        except:
            results.append(-1)
        
        
        
    return results
    
#Função para obter a imagem da grade anotada com os respectivos resultados para cada bola de resposta
def get_images_with_answers(region,vertices,results):
    img_with_answers = region
    count = 0
    
    for i in range(n_grade_lines):
        for j in range(n_grade_cols):
            vert = vertices[i][j]
            res = results[count]
            count+= 1
            
            if res == 0:
                color = (255,0,0)
            elif res == 1:
                color = (0,0,255)
            else :
                color = (255,0,255)
            cv2.putText(img=img_with_answers, text=str(res),org=(vert[0]+5,vert[1]+5), fontFace=4, fontScale=0.5, color=color,thickness=2)

        
    return img_with_answers


#Função para criação do mapa de vértices de acordo com o template utilizado
def create_vertices(first_center,x_space,y_space,region_space,n_region):
    grade = []
    region = []
    question = []
    for n in range(n_region):
        for i in range(10):
            for j in range(4):
                bubble = (first_center[0]+(region_space*n)+(x_space*j),first_center[1]+(y_space*i))
                question.append(bubble)
            region.append(question)
            question = []
        grade.append(region)
        region = []
    
    return grade

#Função que recebe uma imagem e um modelo de detecção e retorna predições e resultados
def get_info(img,detection_model):
    #Cria a array de vértices - arbitrário obtido do template criado
    grade_vertices = create_vertices((first_center_x,first_center_y),x_bubble_space,y_bubble_space,region_space,n_grade_regions)

    #Inicia a imagem e realiza as predições
    img = cv2.resize(img, answer_card_shape)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor,detection_model)
    preds = get_preds(detections, 0.9)

    #Obtém a grade
    box = get_box_by_label(preds,1)
    result = get_object(img,box,0)

    #Obtém o número de inscrição
    insc_box = get_box_by_label(preds,0)
    insc_result = get_object(img,insc_box,0)


    #--------------Obtenção do cartão resposta------------------------------
    area = grade_area
    morph_result = morphology_operations(result)
    rectangles = get_rectangles(morph_result,area)
    result_rectangle = get_result_rectangle(rectangles)
    
    #Obtém a grade e a inclinação da página
    result_grade,theta = get_grade(result,result_rectangle)

    result_grade = cv2.resize(result_grade,grade_shape)    
    result_grade = cv2.cvtColor(result_grade,cv2.COLOR_BGR2GRAY)
    _,result_grade = cv2.threshold(result_grade,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result_grade_BGR = cv2.cvtColor(result_grade.copy(),cv2.COLOR_GRAY2BGR)

    answers = []
    for vertices in grade_vertices:
        result_rois = centralize_rois(result_grade,vertices,bubble_diameter,x_bubble_space)
        results = get_results(result_rois)
        answers.append(results)
        result_img = get_images_with_answers(result_grade_BGR,vertices,results)
    
    #--------------Fim da obtenção do cartão resposta-----------------------------

    #----------------Obtenção do número de inscrição-----------------------------
    insc_gray = cv2.cvtColor(insc_result, cv2.COLOR_BGR2GRAY)
    shape = insc_gray.shape
    new_shape = (shape[1]*2,shape[0]*2)
    insc_img = cv2.resize(insc_gray,new_shape)
    options = "outputbase digits"
    insc_value = pytesseract.image_to_string(insc_img,config=options)
    #---------------Fim da obtenção do número de inscrição----------------------
    
    return result_img,insc_value,answers



#Obtenção das respostas do cartão e geração de um relatório
def get_resp(answer):
    resp = ''
    answered  = False
    for idx,asw in enumerate(answer):
        if asw == -1:
            return 'X' #BRANCO
        elif asw == 0:
            continue
        elif asw == 1 and answered == False:
            answered = True
            index = idx
        elif asw == 1 and answered == True:
            return 'X' #NULL
    
    if answered == False:
        return '- ' #BRANCO
    
    if index == 0:
        return 'A'
    elif index == 1:
        return 'B'
    elif index == 2:
        return 'C'
    elif index == 3:
        return 'D'



#Cria uma dicionario com as respostas de cada questão
def get_answers(answers,lin,col):
    n_regions = len(answers)
    
    results = {}
    count = 1
    
    for r in range(n_regions):
        for i in range(lin):
            answer = []
            for j in range(col):
                asw = answers[r][i*n_grade_cols+j]
                answer.append(asw)
            
            resp = get_resp(answer)
            results[count] = resp
            count += 1
    return results


#Anota as respostas das imagens
def get_image_with_results(img,results_dict):
    grade_vertices = create_vertices((first_center_x,first_center_y),x_bubble_space,y_bubble_space,region_space,n_grade_regions)
    img_with_results = img.copy()
    count = 1
    for vertices in grade_vertices:
        for i in range(10):
            vert = vertices[i][0]
            res = results_dict[count]
            count += 1
            color = (64,0,0)
            cv2.putText(img=img_with_results, text=str(res),org=(vert[0]-20,vert[1]+10), fontFace=4, fontScale=0.5, color=color,thickness=1)
            
    return img_with_results


def create_file_from_dict(answers_dict,dest_path):
    keys = list(answers_dict.keys())
    with open(dest_path,'w') as f:
        for key in keys:
            question = str(key)
            f.write(question + ';' + answers_dict[key] + '\n')
            
    
    