'''
Created on Giu 20, 2018
@author: Giorgio Gosti
'''


#import random
#import math
import time
import matplotlib.pyplot as plt
import matplotlib
import learningRNN as lrnn


import sys, os
import pandas as pd
import networkx as nx
import numpy as np

print 'numpy ver', np.__version__

dataFolder = '' #add folder were dato should be stored if different from current

global glob
glob=False

def Autapsi(a):
    print "Use A for autapsi, NA for no autapsi\n"
    global glob
    if a=="A":
        glob=True
    else:
        glob=False
    print "Autapsi=", glob

def diag_null(C, autapsi=glob): #funzione per annullare le diagonali anche su matrici non quadrate
    if glob==False:
        tall= bool(C.shape[1]<C.shape[0])
        if tall==True:
            C=C.T
        for diag in range(C.shape[1]/C.shape[0]):
            np.fill_diagonal(C[:,C.shape[0]*diag:], 0)
        if tall==True:
            C=C.T
    return C


def transPy(sigma_path0,net1,N,typ = 1, thr = 0):     #non e' stato cambiato
    """
    transiton function. net1 is the network that generates the ttransitions
    
    If sigma_path0 is a binary vector it generates the corresponding transtions.
    
    If sigma_path0 is a list of binary vectors it generates a list with the corresponding transtions.
    
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    if not net1 == np.float32:
        net1 = np.float32(net1)
    if not sigma_path0 == np.float32:
        sigma_path0 = np.float32(sigma_path0)
    sigma_path1 = net1.dot(sigma_path0.T)
    #print sigma_path1
    sigma_path1 [sigma_path1  == 0] = -0.000001
    #print sigma_path1
    sigma_path1 = (1-typ+np.sign(sigma_path1 +thr))/(2-typ)
    #print sigma_path1
    return sigma_path1.T   

#---------------------------------------------------------------------------------
# gradient descent functions
#----------------------------------------------------------------------------------

def gradientDescentStep(y,X,net0,netPars):            #l'unica modifica sui gradient descent e il metodo per azzerare le diagonali
    """
    gradient descent step for the linear approximation of the activation function gradient
    """
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    yhat = transPy(X, net0, N, typ, thr)
    #print 'yhat',yhat
    delta = (y-yhat)
    #print delta
    #print X
    #Xp = np.delete(X, (i), axis=1)
    update = np.asmatrix(X).T.dot(np.asmatrix(delta))
    #print update
    update=diag_null(update)
    if not np.isfinite(np.sum(delta**2)):
        print 'net0'
        print net0
        print 'yhat'
        print yhat
        print 'delta'
        print delta
    return update,np.sum(delta**2),delta,X

def gradientDescentLogisticStep(y,X,k,net0,netPars):
    """
    gradient descent step  for the logistic approximation of the activation function
    """
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    yhat = transPy(X, net0, N, typ, thr)
    #print 'yhat',yhat
    delta = (y-yhat)
    #print delta
    #print 'X'
    #print X
    gamma = np.asarray(np.asmatrix(net0).dot(np.asmatrix(X).T))
    logisticDer = k*np.exp(-k*gamma)/(1+np.exp(-k*gamma))**2
    #print 'logisticDer'
    #print logisticDer.T
    #print X*logisticDer.T
    update = np.asmatrix(X*logisticDer.T).T.dot(np.asmatrix(delta))
    #print update
    update=diag_null(update)
    return update,np.sum(delta**2),delta,X,logisticDer,gamma

def gradientDescentStepDeltaRule(y,X,alpha,net0,netPars):
    """
    gradient descent step
    """
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    yhat = transPy(X, net0, N, typ, thr)
    print 'yhat',yhat
    delta = (y-yhat)
    print delta
    print 'X',X
    dR = net0.dot(X.T)
    print 'dR',dR
    deltaRule = (alpha*np.ones((N,N),dtype=np.float32) >= np.abs(dR))
    print 'deltaRule',deltaRule.T
    #Xp = np.delete(X, (i), axis=1)
    update = np.asmatrix(X).T.dot(np.asmatrix(delta)) 
    #print 'update',update.T
    update = update*deltaRule
    update=diag_null(update)
    return update,np.sum(delta**2)

def stochasticGradientDescentStep(y,X,net0,batchSize,netPars):
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    draws = np.random.choice(y.shape[0],size=batchSize,replace=False)
    ybatch = y[draws,:]
    Xbatch = X[draws,:]
    yhat = transPy(Xbatch, net0, N, typ, thr)
    #yhat = np.array([yhat[:,i]]).T
    delta = (ybatch-yhat)
    #Xp = np.delete(X, (i), axis=1)
    update = Xbatch.T.dot(delta) 
    update=diag_null(update)
    return update,np.sum(delta**2)


# makeTrainXYfromSeqs e' la funzione piu modificata. Di base "incolla" i tempi t-1 sotto i tempi t in X_train, creando vettori piu' alti (nello specifico, alti times*45 neuroni)
def makeTrainXYfromSeqs(seqs,nP, isIndex=True): 
    listX = []
    listy = []
    o_sigma_path = seqs[0]
    times = nP["times"]
    for i in range(times, 0, -1):
        j=times-i
        listX.append( np.array(o_sigma_path[j:-i,:]) )
    X = np.concatenate(listX, axis=1)
    y = np.array(o_sigma_path[times:,:])
    return X,y

def Test(net0, test_set, nP):  # per testare una matrice su un certo set
    errori_test=[]
    for k in range(len(test_set)):
        X_train, Y_train = makeTrainXYfromSeqs([ test_set[k] ], nP, isIndex= False) 
        Y_test= transPy(X_train, net0, nP["N"], nP["typ"], nP["thr"])
        errori_test.append(np.sum((Y_test-Y_train)**2/(nP["N"]*len(Y_train))))
    return errori_test

def NewTest(net0, test_set, nP, t=1, verbose=False):
    errori_guess=[]
    for k in range(len(test_set)):
        X_true, Y_true = makeTrainXYfromSeqs([ test_set[k] ], nP, isIndex= False) 
        Y_guess= transPy(X_true, net0, nP["N"], nP["typ"], nP["thr"])
        
        tot =  Y_true.shape[0]*Y_true.shape[1]
        vp = np.sum(Y_true[Y_true==Y_guess]) / tot               # sommo tutti gli 1 in comune
        vn = -np.sum(Y_true[Y_true==Y_guess]-1) / tot            # sommo tutti gli 0 in comune ( -(0-1)  )
        fp = np.sum(Y_guess[Y_guess>Y_true]) / tot               # sommo tutti gli 1 in guess che non sono in true
        fn = np.sum(Y_true[Y_true>Y_guess]) /tot                 # sommo tutti gli 1 in true che non sono in guess
        
        errori_guess.append([[vp, vn], [fp, fn]])
        if verbose==True: 
            print " Veri positivi= ", np.round(vp, 3), "Veri negativi= ", np.round(vn, 3)
            print "\n Falsi positivi=", np.round(fp, 3), "Falsi negativi=", np.round(fn, 3)
    return errori_guess

def FullTest(net0, test_set, nP, verbose=True, over_verbose=False):
    vfpn_guess=[]
    error=[]
    transition_guess=[]
    transition_error=[]
    for k in range(len(test_set)):
        X_true, Y_true = makeTrainXYfromSeqs([ test_set[k] ], nP, isIndex= False) 
        Y_guess= transPy(X_true, net0, nP["N"], nP["typ"], nP["thr"])
        if nP["times"]>1:                            #per avere le dimensioni giuste a tempi maggiori di 1
            X_true, Y_true = makeTrainXYfromSeqs([ test_set[k][nP["times"]-1:] ], 
                                                      {"N":nP["N"], "typ":nP["typ"], "thr": nP["thr"], "times": 1}, isIndex= False)
        """
            X_true=Y_true=test_set[k]
            for i in range(nP["times"]):
                X_true, Y_temp = makeTrainXYfromSeqs([ X_true ], 
                                                     {"N":nP["N"], "typ":nP["typ"], "thr": nP["thr"], "times": 1}, isIndex= False)
                X_temp, Y_true = makeTrainXYfromSeqs([ Y_true ], 
                                                     {"N":nP["N"], "typ":nP["typ"], "thr": nP["thr"], "times": 1}, isIndex= False)
        """    
        #----------------------------
        errore=(np.sum((Y_true-Y_guess)**2/(nP["N"]*len(Y_guess))))
        #----------------------------
        
        tot =  Y_true.shape[0]*Y_true.shape[1]
        vp = np.sum(Y_true[Y_true==Y_guess]) / tot               # sommo tutti gli 1 in comune
        vn = -np.sum(Y_true[Y_true==Y_guess]-1) / tot            # sommo tutti gli 0 in comune ( -(0-1)  )
        fp = np.sum(Y_guess[Y_guess>Y_true]) / tot               # sommo tutti gli 1 in guess che non sono in true
        fn = np.sum(Y_true[Y_true>Y_guess]) /tot                 # sommo tutti gli 1 in true che non sono in guess
        #----------------------------
        predicted_change = np.array(Y_guess-X_true)
        real_change = np.array(Y_true-X_true)
        good_change = real_change[real_change==predicted_change]
        bad_change = predicted_change[real_change!=predicted_change]
        good_perm = -(good_change**2)+1
        
        guessed_transition = np.sum(good_change**2)/np.einsum("ij->", real_change**2)      # transizioni che avvengono realmente azzeccate
        guessed_permamence = np.sum(good_perm)/np.einsum("ij->", -real_change**2+1)        # permanenze che avvengono realmente azzeccate
        usefull_transition = np.sum(good_change**2)/np.einsum("ij->", predicted_change**2) # transizioni previste vere
        usefull_permanence = np.sum(good_perm)/np.einsum("ij->", -predicted_change**2+1)   # non transizioni previste vere
        
        v_trans_p = np.sum(good_change[good_change==1])/np.sum(real_change[real_change==1])
        v_trans_n = np.sum(good_change[good_change==-1])/np.sum(real_change[real_change==-1])
        f_trans_p = len(bad_change[bad_change==1])/ np.sum(predicted_change[predicted_change==1])
        f_trans_n = -len(bad_change[bad_change==-1])/ np.sum(predicted_change[predicted_change==-1])
        #-----------------------------
        vfpn_guess.append([[vp, vn], [fp, fn]])
        error.append(errore)
        transition_guess.append([[guessed_transition, guessed_permamence],[usefull_transition, usefull_permanence]])
        transition_error.append([[v_trans_p, v_trans_n], [f_trans_p, f_trans_n]])
        if over_verbose==True: 
            print "Errore=", np.round(errore, 5)
            print " Veri positivi= ", np.round(vp, 5), "Veri negativi= ", np.round(vn, 5)
            print "\n Falsi positivi=", np.round(fp, 5), "Falsi negativi=", np.round(fn, 5)
            
    if verbose==True:
        print "\n Errore=", np.round(np.mean(np.array(error)), 5)
        print "\n Veri positivi= ", np.round(np.mean(np.array(vfpn_guess),axis=0)[0,0], 5) 
        print " Veri negativi= ", np.round(np.mean(np.array(vfpn_guess),axis=0)[0,1], 5)
        print " Falsi positivi=", np.round(np.mean(np.array(vfpn_guess),axis=0)[1,0], 5)
        print " Falsi negativi=", np.round(np.mean(np.array(vfpn_guess),axis=0)[1,1], 5)
        print "\n Transizioni realmente avvenute indovinate=", np.round(np.mean(np.array(transition_guess),axis=0)[0,0], 5)
        print " Permanenze realmente avvenute indovinate=", np.round(np.mean(np.array(transition_guess),axis=0)[0,1], 5)
        print " Transizioni previste realmente avvenute=", np.round(np.mean(np.array(transition_guess),axis=0)[1,0], 5)
        print " Permamenze previste realmente avvenute=", np.round(np.mean(np.array(transition_guess),axis=0)[1,1], 5)
        print " Transizioni da 0 a 1 azzeccate su quelle reali=", np.round(np.mean(np.array(transition_error),axis=0)[0,0], 5)
        print " Transizioni da 1 a 0 azzeccate su quelle reali=", np.round(np.mean(np.array(transition_error),axis=0)[0,1], 5)
        print " Transizioni da 0 a 1 sbagliate su quelle previste=", np.round(np.mean(np.array(transition_error),axis=0)[1,0], 5)
        print " Transizioni da 1 a 0 sbagliate su quelle previste=", np.round(np.mean(np.array(transition_error),axis=0)[1,1], 5)
        print "\n Sensibility =", (np.round(np.mean(np.array(vfpn_guess),axis=0)[0,0], 5))/(np.round(np.mean(np.array(vfpn_guess),axis=0)[0,0], 5)+np.round(np.mean(np.array(vfpn_guess),axis=0)[1,1], 5))#vp/p
        print " Specificity=",(np.round(np.mean(np.array(vfpn_guess),axis=0)[1,0], 5))/(np.round(np.mean(np.array(vfpn_guess),axis=0)[1,0], 5)+np.round(np.mean(np.array(vfpn_guess),axis=0)[0,1], 5))#fp/n
        print " Accuracy=", (np.round(np.mean(np.array(vfpn_guess),axis=0)[0,1], 5)+np.round(np.mean(np.array(vfpn_guess),axis=0)[0,0], 5))/(np.round(np.mean(np.array(vfpn_guess),axis=0)[0,0], 5)+np.round(np.mean(np.array(vfpn_guess),axis=0)[0,1], 5)+np.round(np.mean(np.array(vfpn_guess),axis=0)[1,0], 5)+np.round(np.mean(np.array(vfpn_guess),axis=0)[1,1], 5))
    return error, vfpn_guess, transition_guess

def NormMatrici(matrice):
    for i in range(len(matrice)): 
        matrice[i]= matrice[i]/(np.dot(matrice[i], matrice[i]))**(0.5) #normalizzo ogni riga di Jij        
    matrice /= (matrice.shape[0])**(0.5)
    return matrice

# le uniche modifiche sono che ora la funzione si calcola un parametro M per l'altezza delle matrici e il metodo per annullare le diagonali.
def runGradientDescent(X,y,alpha0, net=[], alphaHat=None, nullConstr = None,batchFr = 10.0, passi=10**6, runSeed=3098, gdStrat='SGD', k=1,netPars={'typ':0.0},showGradStep=1, verbose = True, xi = 0 ,uniqueRow=False,lbd = 0.0,mexpon=-1.8):
    N= X.shape[1]
    M= y.shape[1]
    np.random.seed(runSeed)
    if net==[]:
        net0 = np.float32(2*np.random.rand(M,N)-1) #np.zeros((r, w), dtype=np.float32)  # np.float32(np.random.randint(0, 2, size=(r, w)))  # np.float32(2*np.random.rand(r,w)-1)
    else: net0=net
    net0=diag_null(net0)
    
    #net0 = lrnn.rowNorm(net0)
    if not nullConstr == None: net0[nullConstr==True]=0
     
    #print 'start net0'
    #print net0
    #print np.sum(np.abs(net0),axis=1)
    m = X.shape[0]
    if verbose: print 'm ',m
    if uniqueRow == True:
        new_array = [''.join( str(e) for e in np.uint8(row).tolist() ) for row in X]
        Xunique, index = np.unique(new_array,return_index=True)
        X = X[index,:]
        y = y[index,:]
        m = X.shape[0]
        if verbose: print 'm unique ',m
        plt.figure()
        plt.imshow(X,interpolation='nearest')
        plt.figure()
        plt.imshow(np.corrcoef(X.T),interpolation='nearest')
        plt.colorbar()
    if not gdStrat == 'SGD': batchFr = 1.0
    batchSize = m/batchFr
    if verbose: print 'batchSize',batchSize,'fract',batchFr
    if alpha0 == 0.0: alpha0 =alphaHat *( m **(-1.0) ) *( N **(mexpon) ) #alpha0 =alphaHat /  ( m *  N**2)
    if verbose: print 'alphaHat',alphaHat,'alpha0',alpha0
        
    convStep = np.inf
    deltas = []
    fullDeltas = []
    start = time.time()
    for j in xrange(passi):
        alpha = alpha0* ( (1+alpha0*lbd*j)**(-1))    
        if gdStrat == 'SGD':
            update,sumSqrDelta = stochasticGradientDescentStep(y,X,net0,batchSize,netPars)
        elif gdStrat == 'GD':
            update,sumSqrDelta,delta,X = gradientDescentStep(y,X,net0,netPars)
            if not np.isfinite(sumSqrDelta):
                break
        elif gdStrat == 'GDLogistic':
            update,sumSqrDelta,delta,X,logisticDer,gamma = gradientDescentLogisticStep(y,X,k,net0,netPars)
            print update
        if j%(passi/200) == 0:
            #print j
            #print 'yhat '
            #print yhat,yhat.shape
            #print 'sumSqrDelta ', sumSqrDelta
            fullSumSqrDelta = sumSqrDelta
            if batchFr < 1.0: updatefull,fullSumSqrDelta,delta,X = gradientDescentStep(y,X,net0,netPars)

            # qui ho tolto showgradstep

            deltas.append(sumSqrDelta/batchSize)
            fullDeltas.append(fullSumSqrDelta/y.shape[0])
        if sumSqrDelta == 0.0:
            fullSumSqrDelta = 0
            if batchFr < 1.0:
                updatefull,fullSumSqrDelta,delta,X = gradientDescentStep(y,X,net0,netPars)
            if fullSumSqrDelta == 0:
                deltas.append(sumSqrDelta/batchSize)
                fullDeltas.append(fullSumSqrDelta/y.shape[0])
                convStep = j
                if verbose: print 'final sumSqrDelta/batchSize ', sumSqrDelta/batchSize
                break
        #print 'sparce'
        #print net0
        #print xi * net0
        net0 += alpha * update.T #- xi * (net0 / net0.mean())
        if not nullConstr == None: net0[nullConstr==True]=0
        #net0[net0>1] = 1
        #net0[net0<-1] = -1
        #net0 = NormMatrici(net0)
        net0=lrnn.rowNorm(net0)
        #print 'net0',net0.shape
    if verbose: print 'final sumSqrDelta ', sumSqrDelta,not np.isfinite(sumSqrDelta)
    if verbose: print 'final sumSqrDelta/batchSize ', sumSqrDelta/batchSize
    #if not np.isfinite(sumSqrDelta):
        #print deltas
    end = time.time()
    exTime = end - start
    if verbose: print 'decent time', exTime
    return net0,deltas,fullDeltas,exTime,convStep




















