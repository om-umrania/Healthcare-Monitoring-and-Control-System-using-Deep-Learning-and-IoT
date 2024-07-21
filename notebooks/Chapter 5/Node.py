import math
import random
import CryptographicFunctions as cf

class Node:
    def __init__(self, name, location, trust=0, energy=1000):
        self.name = name
        self.trust = trust
        self.energy = energy
        self.location = location
        self.blockchain = []

    def modifyTrust(self, newTrust):
        self.trust += newTrust
        if self.trust <= 0:
            self.trust = 1
        return True

    def canAddBlock(self, block):
        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(self.blockchain, hash_val)
        return isUnique

    def getTrustScore(self, other_node, check=True):
        loc1 = self.location
        loc2 = other_node.location

        dist = math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
        trust_val = (other_node.energy / (dist + 1)) + (other_node.trust / (self.trust + 1))

        if self.validateBlockchain() and other_node.validateBlockchain():
            return trust_val
        elif check:
            print('\tSelf correcting chains...')
            self.correctBlockchain()
            other_node.correctBlockchain()
            return trust_val
        else:
            return trust_val

    def storeBlockchain(self, blockchain):
        self.blockchain = blockchain
        return True

    def addBlock(self, block):
        self.blockchain.append(block)
        return True

    def attackNode(self, threshold):
        val = random.random()
        if val < threshold:
            index = round(random.random() * (len(self.blockchain) - 1))
            if index < 0:
                return False
            try:
                self.blockchain[index]['prevHash'] = ''
                self.reduceEnergy()
                print(f'\tAttacking node {self.name}')
            except:
                print('\tWill attack in next iteration')
            return True
        else:
            return False

    def correctBlockchain(self):
        for count in range(2, len(self.blockchain)):
            prevHash = self.blockchain[count - 1]['hash']
            self.blockchain[count]['prevHash'] = prevHash

        self.reduceEnergy()
        print(f'\tChain corrected for node {self.name}...')
        return True

    def reduceEnergy(self):
        if self.energy > 0:
            self.energy -= 1
        return True

    def validateBlockchain(self):
        for count in range(2, len(self.blockchain)):
            prevHash = self.blockchain[count - 1]['hash']
            if self.blockchain[count]['prevHash'] != prevHash:
                self.modifyTrust(-1)
                return False

        self.reduceEnergy()
        self.modifyTrust(1)
        return True
