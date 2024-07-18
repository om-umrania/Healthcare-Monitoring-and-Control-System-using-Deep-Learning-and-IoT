from tinyec import registry
from Crypto.Cipher import AES
import hashlib, secrets, binascii
import time
import matplotlib.pyplot as plt

def encrypt_AES_GCM(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ecc_point_to_256_bit_key(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()

def encrypt_ECC(msg, pubKey):
    ciphertextPrivKey = secrets.randbelow(curve.field.n)
    sharedECCKey = ciphertextPrivKey * pubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    ciphertext, nonce, authTag = encrypt_AES_GCM(msg, secretKey)
    ciphertextPubKey = ciphertextPrivKey * curve.g
    return (ciphertext, nonce, authTag, ciphertextPubKey)

def decrypt_ECC(encryptedMsg, privKey):
    (ciphertext, nonce, authTag, ciphertextPubKey) = encryptedMsg
    sharedECCKey = privKey * ciphertextPubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    plaintext = decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey)
    return plaintext

curveList = list(registry.EC_CURVE_REGISTRY.keys())
cName = curveList[0]
curve = registry.get_curve(cName)

def eccEncrypt(msg) :
    
    msg = str(msg['src']) + str(msg['dest']) + str(msg['data']) + str(msg['ts']) + str(msg['nonce']) + str(msg['prevHash'])
    msg = msg.encode('utf-8')
    
    print('Message:%s' % msg)
    
    privKey = secrets.randbelow(curve.field.n)
    pubKey = privKey * curve.g
    
    encryptedMsg = encrypt_ECC(msg, pubKey)
    
    print(encryptedMsg)
    
    return encryptedMsg