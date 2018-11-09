# -*- coding: utf-8 -*-
# Plot accuracy(10-fold val & test) for models
import numpy as np
import matplotlib.pyplot as plt
def plotAcc(usedModelList, title, xlabel='Model', ylabel='Accuracy'):    
    n_groups = len(usedModelList)
    paraCount = len(usedModelList[0])
    names = [item[0] for item in usedModelList]
    CVs   = [item[1] for item in usedModelList]
    scores= [item[2] for item in usedModelList]
    if paraCount ==5:
        stds = [item[4] for item in usedModelList]
    else:
        stds = []
        
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
         
    opacity = 0.4    
    rects1 = plt.bar(index, CVs, bar_width, alpha=opacity, color='b',label='10-fold Cross Validation')
    rects2 = plt.bar(index + bar_width, scores, bar_width, alpha=opacity, color='r', label='Test Accuracy')    
    
    def autolabel(rects):    
    #Attach a text label above each bar displaying its height    
        for rect in rects:
            #height = round(rect.get_height()*10000)/100
            height = rect.get_height()
            print(height)
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    #'%d' % int(height),
                    str(int(height*10000)/100),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)    
    plt.title(title)
    plt.xticks(index + bar_width -0.17, names)    
    plt.legend();
      
    plt.tight_layout();   
    
    
    plt.show();       
    plt.ylim(ymax=1)