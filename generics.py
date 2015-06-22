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

base_img = cv2.imread("assets/matching.png",0)#対象画像読み込み
base_h,base_w = base_img.shape
edge_img = cv2.imread("assets/sobelBinary.png",0)#エッジ画像読み込み
temp_name_list = os.listdir('templates/')
temps = []

for name in temp_name_list:
    print name
    img = Image.open('templates/%s' % name).convert("L")
    temps.append(img)
num_of_temp = len(temps)
temp_h,temp_w = temps[0].size

mask = Image.open("templates/mask.png").convert("L")

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
def fitnessMax(individual):
	#遺伝子をテンプレートマッチングの各パラメータにエンコード	
    t = int("".join(map(str,individual[0:4])),2)
    x = int("".join(map(str,individual[4:13])),2)
    y = int("".join(map(str,individual[13:22])),2)
    k = int("".join(map(str,individual[22:26])),2)
    theta = int("".join(map(str,individual[26:30])),2)
    trans_h = temp_h*k
    trans_w = temp_w*k
    #パラメータに従って適応度を算出
    if (k!=0):        #k=0のときは画像サイズそのまま，k=1=0
        img_size = temps[t].resize((trans_w,trans_h))
        img_size_mask = mask.resize((trans_w,trans_h))
    else:
        img_size = temps[t]
        img_size_mask = mask

    img_rotate = img_size.rotate(theta,expand=1)#回転させる，expand=1=はみ出し部分の拡張
    img_rotate_mask = img_size_mask.rotate(theta,expand=1)
    img_rotate = np.asarray(img_rotate)#pil形式をopencv形式になおす
    img_rotate_mask = np.asarray(img_rotate_mask)
    template_h,template_w = img_rotate.shape#できあがったテンプレート画像のサイズを取得	

 
    if y-template_h/2<0 or x-template_w/2<0:#はみだしを考慮して原画像の一部を切り取る
        compare_img = base_img[0:template_h,0:template_w]
    elif x+template_w/2 > base_w:
        compare_img = base_img[y-template_h/2:(y-template_h/2)+template_h,x-template_w/2:(x-template_w/2)+(template_w-(x+template_w/2-base_w))]
    elif y+template_h/2 > base_h:
        compare_img = base_img[y-template_h/2:(y-template_h/2)+(template_h-(y+template_h/2-base_h)),x-template_w/2:(x-template_w/2)+template_w]
    else:
        compare_img = base_img[y-template_h/2:(y-template_h/2)+template_h,x-template_w/2:(x-template_w/2)+template_w]
    compare_h,compare_w = compare_img.shape

    diff = 0
    for i in range(0,compare_h):#差の和をだす
        for j in range(0,compare_w):
            if img_rotate_mask[i,j]==255:
                diff+=compare_img[i,j]-img_rotate[i,j] if compare_img[i,j]>img_rotate[i,j] else img_rotate[i,j]-compare_img[i,j]

    f = -1*diff/float(255*compare_h*compare_w)#適応度f
    diff = 0
    #エッジ率の計算，ペナルティをかけるくだり
    if y-template_h/2<0 or x-template_w/2<0:#おなじようにはみだしを考慮してエッジ画像の一部を切り取る
        edge = edge_img[0:template_h,0:template_w]
    elif x+template_w/2 > base_w or y+template_h/2 > base_h:
        edge = edge_img[y-template_h/2:(y-template_h/2)+(template_h-(y+template_h/2-base_h)),x-template_w/2:(x-template_w/2)+(template_w-(x+template_w/2-base_w))]         
    else:
        edge = edge_img[y-template_h/2:(y-template_h/2)+template_h,x-template_w/2:(x-template_w/2)+template_w]         

    e_count = 0
    if (k==0):#エッジ成分の割合の閾値eiを算出する
        ei = 1.5
    else:
        ei = 1.5/k
    for i in range(0,compare_h):#白い部分＝エッジ部分の数を数える
        for j in range(0,compare_w):
            if (edge[i,j]==255):
                e_count+=1
    if (ei >= e_count/template_w*template_h):#ペナルティをかける．この計算が合ってるかは疑問	
        f = f*3
    return f,

# Operator registering
toolbox.register("evaluate", fitnessMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=2)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=100)
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
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    t = int("".join(map(str,best_ind[0:4])),2)
    x = int("".join(map(str,best_ind[4:13])),2)
    y = int("".join(map(str,best_ind[13:22])),2)
    k = int("".join(map(str,best_ind[22:26])),2)
    theta = int("".join(map(str,best_ind[26:30])),2)
    print "TempNumber = %d, X = %d, Y = %d, theta = %d" % (t,x,y,theta)

    cv2.rectangle(base_img,(x-((temp_w*k)/2),y-((temp_h*k)/2)),(x+((temp_w*k)/2),y+((temp_h*k)/2)),(0,0,255),5)
    cv2.imwrite("rect.png",base_img)
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
