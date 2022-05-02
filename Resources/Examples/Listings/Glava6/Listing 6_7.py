# Listing 6.7
y = net.activate([1, 1])
print('Y1=', y)

fileObject = open('MyNet.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()

fileObject = open('MyNet.txt', 'rb')
net2 = pickle.load(fileObject)

y = net2.activate([1, 1])
print('Y2=', y)
