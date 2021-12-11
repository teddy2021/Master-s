import Network

net = Network.SegNetwork("./seg_data_files.csv", 1)
results = net.forward()
print(f"After one epoch, the network has {results[1]} loss")
