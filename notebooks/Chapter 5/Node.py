class Node:
    def __init__(self, name, location, trust = 0, energy = 1000):
        self.name = name
        self.trust = trust
        self.energy = energy
        self.location = location
        self.blockchain = []
        
    def modifyTrust(n, newTrust) :
        n.trust = n.trust + newTrust
        if(n.trust <= 0) :
            n.trust = 1
            
        return True
    
    def canAddBlock(n, block) :
        import time
        import random
        import CryptographicFunctions as cf
        
        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(n.blockchain, hash_val)
        return isUnique
    
    def getTrustScore(n1, n2, check = True) :
        import math
        
        loc1 = n1.location
        loc2 = n2.location
        
        x1 = loc1[0]
        y1 = loc1[1]
        x2 = loc2[0]
        y2 = loc2[1]
        
        dist = math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        trust_val = ((n2.energy)/(dist+1)) + (n2.trust)/(n1.trust+1)
        if(n1.validateBlockchain() and n2.validateBlockchain()) :
            return trust_val
        elif(check == True) :
            print('\tSelf correcting chains...')
            n1.correctBlockchain()
            n2.correctBlockchain()
            return trust_val
        else :
            return trust_val
    
    def storeBlockchain(n, blockchain) :
        n.blockchain = blockchain
        return True
    
    def addBlock(n, block) :
        #print('Node %s, blocks %d' % (n.name, len(n.blockchain)))
        n.blockchain.append(block)
        return True
    
    def attackNode(n, threshold) :
        import random
        val = random.random()
        if(val < threshold) :
            index = round(random.random() * (len(n.blockchain)-1))
            if(index < 0) :
                return False
            try :
                n.blockchain[index]['prevHash'] = ''
                n.reduceEnergy()
                print('\tAttacking node %s' % (n.name))
            except :
                print('\tWill attack in next iteration')
                
            return True
        else :
            return False
        
    def correctBlockchain(n) :
        for count in range(2, len(n.blockchain)) :
            
            prevHash = n.blockchain[count-1]['hash']
            n.blockchain[count]['prevHash'] = prevHash
        
        n.reduceEnergy()
        print('\tChain corrected for node %s...' % (n.name))
        return True
    
    def reduceEnergy(n) :
        if(n.energy > 0) :
            n.energy = n.energy - 1
        return True
    
    def validateBlockchain(n) :
        for count in range(2, len(n.blockchain)) :
            
            prevHash = n.blockchain[count-1]['hash']
            if(n.blockchain[count]['prevHash'] != prevHash) :
                n.modifyTrust(-1)
                return False;
        
        n.reduceEnergy()
        n.modifyTrust(1)
        return True