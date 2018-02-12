import random

#generator example
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex<len(train_data):
        batch = train_data[sindex:eindex]
        sindex = eindex
        eindex = eindex + batch_size
        yield batch

    if eindex>len(train_data):
        batch = train_data[sindex:]
        yield batch

