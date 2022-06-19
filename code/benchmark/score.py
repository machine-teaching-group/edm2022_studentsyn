import numpy as np 

def score(instances, labels, preds):

    for idx, instance in enumerate(instances) :         
        _, _, _, options = instance 
        label = labels[idx]
        pred = preds[idx]
        if pred == label : continue 
        # if predicted code is the same as label code 
        # count prediction as correct
        if options[pred] == options[label]:            
            preds[idx] = labels[idx]

    accuracy = len(np.where(labels == preds)[0]) / len(labels)
    return accuracy  
