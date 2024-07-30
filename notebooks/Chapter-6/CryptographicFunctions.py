import hashlib

def findHash(block):
    block_data = str(block['src']) + str(block['dest']) + str(block.get('data', '')) + str(block['ts']) + str(block['nonce']) + str(block['prevHash'])
    return hashlib.sha256(block_data.encode('utf-8')).hexdigest()

def checkForUniqueHash(blockchain, hashVal):
    if len(blockchain) > 1:
        conditions = findCondition(blockchain[0]['nonce'])
        l = len(conditions)
        if hashVal[0:l] != conditions:
            return False
    
    for count in range(1, len(blockchain)):
        if blockchain[count]['hash'] == hashVal:
            return False
    
    return True 

def findCondition(nonce):
    l = len(str(nonce))
    l = int(round((20 - l) / 8))
    val = ""
    for count in range(l):
        val += "0"
    
    return val
