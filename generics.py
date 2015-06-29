# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import os
import random
import math
from PIL import Image
from deap import base
from deap import creator
from deap import tools

p=""
print len(sys.argv)
if 1<len(sys.argv):
    p=sys.argv[1]
base_img = cv2.imread("assets/matching.png",0)#対象画像読み込み
base_h,base_w = base_img.shape
edge_img = cv2.imread("assets/sobelBinary.png",0)#エッジ画像読み込み
temp_name_list = os.listdir('templates/')
temps = []

each_radian = 360/16

for name in temp_name_list:
    print name
    img = Image.open('templates/%s' % name).convert("L")
    temps.append(img)
    if "n" in p:
      break;

num_of_temp = len(temps)
#temp = Image.open('templates/0.png').convert("L")
temp_h,temp_w = temps[0].size
mask_img = Image.open("templates/mask.png").convert("L")
mask_img = mask_img.resize((temp_h,temp_w))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#@profile
def individual_param(individual):
    if "n" in p:
        t=0
    else:
        t = int("".join(map(str,individual[0:4])),2)
    x = int("".join(map(str,individual[4:13])),2)
    y = int("".join(map(str,individual[13:22])),2)
    if "k" in p:
        k=1
    else:
        k = int("".join(map(str,individual[22:26])),2)
    theta = int("".join(map(str,individual[26:30])),2)
    if k==0:
        k=1

    return (t,x,y,k,theta)

def fitnessMax(individual):
#遺伝子をテンプレートマッチングの各パラメータにエンコード 
    t,x,y,k,theta = individual_param(individual);
    trans_h = temp_h*k
    trans_w = temp_w*k
    #パラメータに従って適応度を算出
    img_size = temps[t].resize((trans_w,trans_h))
    img_size_mask = mask_img.resize((trans_w,trans_h))
    if "r" in p:
        img_rotate = img_size
        img_rotate_mask = img_size_mask
    else:
        img_rotate = img_size.rotate(theta*each_radian,expand=1)#回転させる，expand=1=はみ出し部分の拡張
        img_rotate_mask = img_size_mask.rotate(theta*each_radian,expand=1)
    img_rotate = np.asarray(img_rotate)#pil形式をopencv形式になおす
    img_rotate_mask = np.asarray(img_rotate_mask)
    template_h,template_w = img_rotate.shape#できあがったテンプレート画像のサイズを取得  

    absolute_x_start = x-template_w/2 #切り出す原画像の左上位置
    absolute_y_start = y-template_h/2
    absolute_x_end = x+template_w/2 #切り出す原画像の右下位置
    absolute_y_end = y+template_h/2
    compare_img = base_img[absolute_y_start:absolute_y_start+template_h,absolute_x_start:absolute_x_start+template_w]
    compare_img = cv2.resize(compare_img,(template_w,template_h))
    compare_h,compare_w = compare_img.shape

    mask = img_rotate_mask==255
    diff =  img_rotate-compare_img
    np.abs(diff)
    Different = np.sum(diff[mask])
    
    f = -1*Different/float(255*compare_h*compare_w)#適応度f
    #print f
    if "e" in p:
        return f,
    #エッジ率の計算，ペナルティをかけるくだり
    edge = edge_img[absolute_y_start:absolute_y_start+template_h,absolute_x_start:absolute_x_start+template_w]         
    edge = cv2.resize(edge,(template_w,template_h))
    e_count = 0
    ei = 1.5/k

    edge_mask = edge[mask]
    edge_white_mask = edge_mask==255
    e_count = edge_mask[edge_white_mask]
    e_count = e_count.size

    if e_count == 0:
        e_count = 1
    if (ei >= e_count/template_w*template_h):#ペナルティをかける．この計算が合ってるかは疑問 
        print "penalty"
        f = f*3
    return f,
def is_can_use(individual):
    t,x,y,k,theta = individual_param(individual)

    trans_h = temp_h*k
    trans_w = temp_w*k
    img_size = temps[t].resize((trans_w,trans_h))
    if "r" in p:
        img_rotate = img_size
    else:
        img_rotate = img_size.rotate(theta*each_radian,expand=1)#回転させる，expand=1=はみ出し部分の拡
    img_rotate = np.asarray(img_rotate)#pil形式をopencv形式になおす
    template_h,template_w = img_rotate.shape#できあがったテンプレート画像のサイズを取得  
    absolute_x_start = x- template_w/2 #切り出す原画像の左上位置
    absolute_y_start = y-template_h/2
    absolute_x_end = x+template_w/2 #切り出す原画像の右下位置
    absolute_y_end = y+template_h/2
    if absolute_y_start<=0 or absolute_x_start<=0 or absolute_x_end >= base_w or absolute_y_end >= base_h:
        return False
    else:
        return True

def mute_orverride(child1,child2):
    while True:
        a,b = tools.cxOnePoint(child1,child2)
        if is_can_use(a) and is_can_use(b):
            return (a,b)
def mutate_override(mutant):
    while True:
        indivi = tools.mutFlipBit(mutant,indpb=0.05)
        if is_can_use(indivi[0])==True and indivi[0] != None:
            return indivi[0]

def individual_override():
    while True:
        indivi = toolbox.individual()
        if is_can_use(indivi)==True and indivi != None:
            return indivi

def population_override(n):
    pop = []
    for i in range(0,n):
        a = individual_override()
        pop.append(a)
    return pop


# Operator registering
toolbox.register("evaluate", fitnessMax)
toolbox.register("mate", mute_orverride)
toolbox.register("mutate", mutate_override)
toolbox.register("select", tools.selTournament, tournsize=2)

def print_img_window(individual,gen):
    t,x,y,k,theta = individual_param(individual)

    print "TempNumber = %d, X = %d, Y = %d, theta = %d" % (t,x,y,theta)

    trans_h = temp_h*k
    trans_w = temp_w*k
    absolute_x_start = x- trans_w/2 #切り出す原画像の左上位置
    absolute_y_start = y-trans_h/2
    absolute_x_end = x+trans_w/2 #切り出す原画像の右下位置
    absolute_y_end = y+trans_h/2

    img_size = temps[t].resize((trans_w,trans_h))
    if "r" in p:
        img_rotate = img_size
    else:
        img_rotate = img_size.rotate(theta*each_radian,expand=1)#回転させる，expand=1=はみ出し部分の拡張
    img_rotate = np.asarray(img_rotate)#pil形式をopencv形式になおす
    rect_img = base_img.copy()
    cv2.rectangle(rect_img,(absolute_x_start,absolute_y_start),(absolute_x_end,absolute_y_end),(0,0,255),2)
    cv2.imwrite("result/rect%d.png"%gen,rect_img)

    cv2.imshow("base Generation %d"%gen,rect_img)
    cv2.imshow("temp Generation %d"%gen,img_rotate)
    #cv2.waitKey(0)                  # キー入力待機
    if gen!=0:
        cv2.destroyWindow("temp Generation %d"%(gen-1))
        cv2.destroyWindow("base Generation %d"%(gen-1)) 

def main():
    random.seed(64)
    
    pop = population_override(n=100)
    CXPB, MUTPB, NGEN = 0.7, 0.05, 100
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print_img_window(ind,g)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print_img_window(best_ind,NGEN)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
