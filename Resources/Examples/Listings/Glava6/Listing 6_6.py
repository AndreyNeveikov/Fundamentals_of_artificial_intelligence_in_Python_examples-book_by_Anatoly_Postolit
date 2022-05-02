# Listing 6.6
fileObject = open('MyNet.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()
fileObject = open('MyNet.txt', 'rb'))
net2 = pickle.load(fileObject)
