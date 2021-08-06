#Quantum Mining
#Explorer: https://www.blockchain.com/explorer

import hashlib
m = hashlib.sha256()

# Merkle
# 93d1ce9f7b3d41013dd898b8da2f744479a91b0be11ae2263b4c01ae918ec110
m.update(b"93d1ce9f7b3d41013dd898b8da2f744479a91b0be11ae2263b4c01ae918ec110")
m.digest()

#b'\x18\xac\xde\xda\x0c\xd1\xbe:7\x98t\xd2\x11\xc1\x98\x1b\xfa\xbbU\xc2o\xdc:\xac\xad\xe46\x15\x9eR\x17\xe0'
#000000000000000000068983c42ee2e725ea9f07a2ec71ea7d7b2df133077ee2