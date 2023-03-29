import json
import os
import redis
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, request, Response

from BlackScholes import BlackScholes
from PricingModel import *
from flask_cors import cross_origin
from threading import Thread
from queue import Queue

app = Flask("Pricing Application")

instrumentModelMap = {}
size = 8192
seed = np.random.randint(0, 10000)

trainingDataAPIUrl = "http://127.0.0.1:8092"

@app.route('/', methods=['GET','POST'])
@cross_origin()
def welcome():
    return "Welcome to Option Pricing using Differential AAD Application "

@app.route('/init', methods=['GET','POST'])
@cross_origin()
def initPricingApp():
    print('Init End point : Initializing the Pricing App')
    instrumentList = fetchAllInstruments()
    totalInstrument = len(instrumentList)

    if (totalInstrument == 0):
        return Response('Could not fetch instrument List from Redis cache')

    redis_con = connectToRedis()
    count = 0
    response = {'modelDetails':[]}
    for instrument in instrumentList:
        print(f'Initiating Model Creation for {instrument}')
        try:
            # Fetch Training data from redis
            if modelExist(instrument):
                print(f'Model Exist for instrument = {instrument} ,Not creating again ')
                continue

            training_data = json.loads(redis_con.get(instrument))
            model, trainingTime = initPriceModelforGivenInstrument(instrument,training_data)
            print(f'Model initialized for instrument = {instrument} with trainingTime = {trainingTime}')
            populateModelCache(instrument, model)
            response['modelDetails'].append({'instrumentId':instrument , 'status':'Model Initialized Successfully'})
        except Exception as e:
            print('Exception occured while getting training data and initializing Model', str(e))
            error_message = 'Could not determine price' + ' cause : ' + str(e)
            response['modelDetails'].append({'instrumentId': instrument, 'status': error_message})
        finally:
            count = count + 1
            print(f'*** Instrument Processed so far = {count}')

    response['Total Instrument'] = totalInstrument
    response['Failed Instrument'] = totalInstrument - count
    response['Initialized Instrument'] = count
    return jsonify(response)


@app.route('/init/parallel', methods=['GET','POST'])
@cross_origin()
def initPricingApp1():
    print('Init End point : Initializing the Pricing App in parallel mode')
    ts = time.time()
    instrumentList = fetchAllInstruments()
    totalInstrument = len(instrumentList)

    if (totalInstrument == 0):
        return Response('Could not fetch instrument List from Redis cache')

    redis_con = connectToRedis()
    count = 0
    response = {'modelDetails':[]}

    global sharedData
    sharedData = Metadata(count,response)
    queue = Queue()

    no_of_workers = 5
    no_of_workers_str = os.environ.get('NO_OF_THREADS')
    if no_of_workers_str:
        no_of_workers = int(no_of_workers_str)

    print(f'Spawning {no_of_workers} worker Threads')
    for x in range(no_of_workers):
        worker = PricingModelWorker('worker-'+str(x),queue, sharedData)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()

    for instrument in instrumentList:
        print(f'Queueing {instrument} for price worker to process')
        queue.put(instrument)

    queue.join()


    response['Total Instrument'] = totalInstrument
    response['Failed Instrument'] = totalInstrument - len(instrumentModelMap)
    response['Initialized Instrument'] = len(instrumentModelMap)

    print('Complete initialization Took ', time.time() - ts)

    return jsonify(response)



class PricingModelWorker(Thread):
    def __init__(self, threadName ,queue, metadata):
        Thread.__init__(self,name=threadName)
        self.queue = queue
        self.metadata = metadata

    def run(self):

        while True:
            # Get the instrument from queue and initialize the model
            threadId = self.name
            instrument = self.queue.get()
            print(f'{threadId} - Initiating Model Creation for {instrument}')
            try:
                # Fetch Training data from redis
                if modelExist(instrument):
                    print(f'{threadId} - Model Exist for instrument = {instrument} ,Not creating again ')
                    continue

                training_data = json.loads(connectToRedis().get(instrument))
                model, trainingTime = initPriceModelforGivenInstrument(instrument, training_data)
                print(f'{threadId} - Model initialized for instrument = {instrument} with trainingTime = {trainingTime}')
                populateModelCache(instrument, model)

                self.metadata.response['modelDetails'].append(
                    {'{threadId} - instrumentId': instrument, 'status': 'Model Initialized Successfully'})
            except Exception as e:
                print('Exception occured while getting training data and initializing Model', str(e))
                error_message = 'Could not determine price' + ' cause : ' + str(e)
                self.metadata.response['modelDetails'].append({'instrumentId': instrument, 'status': error_message})
            finally:
                self.queue.task_done()

class Metadata:
    def __init__(self, count, response):
        self.instrumentCount = count
        self.response = response


@app.route('/model/price/instrument', methods=['POST'])
@cross_origin()
def getPredictedPriceForInstrument():
    if (request.method == 'POST'):
        request_data = request.get_json()
        instrumentId = request_data['instrumentId']
        spotPrice = request_data['spotprice']

        response = getRawResponse()
        trainingTime = float(0)
        # First Check the pricer modle in cache
        pricerModel = instrumentModelMap.get(instrumentId)

        if not pricerModel:
            print("No model found for given Instrument in Model Cache, Iniiating the model inititilization ")
            try:
                redis_con = connectToRedis()
                training_data = redis_con.get(instrumentId)
                print(training_data)
                model,trainingTime = initPriceModelforGivenInstrument(instrumentId,training_data)
                print('training time = {trainingTime}')
                populateModelCache(instrumentId,model)
            except Exception as e:
                print('Exception occured while getting training data and initializing Model',str(e))
                error_message = 'Could not determine price'+' cause : '+str(e)
                return Response(error_message,400)


        pricerModel = instrumentModelMap.get(instrumentId)
        if not pricerModel:
            print('Could not create Model for given instrument')
            response["data"].append({'instrumentId': instrumentId, 'values': ''})

        else:
            spotPriceArr = (np.array(spotPrice)/100).reshape([-1, 1])
            predictions, deltas , prediction_time = predictPriceForInstrument(instrumentId , spotPriceArr, pricerModel)
            #predictions = predictions * 100
            trainingTime = 0 if not trainingTime else trainingTime
            totaltimetaken = trainingTime+prediction_time
            print(f'total time taken = {totaltimetaken}')
            original_spotPrice = np.array(spotPrice).reshape([-1,1])
            values = np.concatenate((original_spotPrice, predictions, deltas), axis=1)
            df = pd.DataFrame(values, columns=['spotPrice', 'predictedPrice', 'delta'])
            enrichResponse(response,instrumentId,prediction_time, trainingTime, totaltimetaken, df.to_dict(orient="records"))

        return jsonify(response)

    
    

@app.route('/model/price/instrument1', methods=['POST'])
@cross_origin()
def getPredictedPriceForInstrument1():
    if (request.method == 'POST'):
        print('Hitting the Instrument 1 API ')
        request_data = request.get_json()
        instrumentId, strikePrice, expiryInYears, spotPrice, volatality = getRequestParam(request_data)

        # simulation set sizes to perform
        sizes = [8192]
        simulSeed = np.random.randint(0, 10000)
        print("using seed %d" % simulSeed)
        weightSeed = None

        # number of test scenarios
        nTest = 100

        generator = BlackScholes()
        generator.__init__(spot=spotPrice, K=strikePrice, vol=volatality, T2=expiryInYears)

        xAxis, yTest, dydxTest, vegas, values, deltas = test(generator, sizes, nTest, simulSeed, None, weightSeed,
                                                                 instrumentId=instrumentId)

        bsspot = np.array(xAxis).reshape([-1,1])
        bsValue = np.array(yTest).reshape([-1, 1])
        modelPredictions = np.array(values[('differential', 8192)]).reshape([-1, 1])


        price_data = np.concatenate((bsspot, bsValue, modelPredictions), axis=1)
        df = pd.DataFrame(price_data, columns=['spot', 'bsprice', 'modelprice'])
        price_data_dict = df.to_dict(orient="records")
        #
        print(price_data_dict)

        return jsonify(price_data_dict)
    
    

def test(generator,
         sizes,
         nTest,
         simulSeed=None,
         testSeed=None,
         weightSeed=None,
         deltidx=0,
         instrumentId=None):
    # simulation
    print("simulating training, valid and test sets")
    xTrain, yTrain, dydxTrain = generator.trainingSet(max(sizes), seed=simulSeed)

    comb_arr = np.concatenate((xTrain,yTrain,dydxTrain), axis = 1)

    xTest, xAxis, yTest, dydxTest, vegas = generator.testSet(num=nTest, seed=testSeed,spotvariation=0.65)
    print("done")

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print("done")

    predvalues = {}
    preddeltas = {}
    for size in sizes:
        print("\nsize %d" % size)
        print('Model prep start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
        regressor.prepare(size, False, weight_seed=weightSeed)
        print('model prep end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

        t0 = time.time()
        print('*** Started Training time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
        regressor.train("standard training")

        print('*** Training Finished , time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

        instrumentModelMap.__setitem__(instrumentId,regressor)

        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()

        regressor.prepare(size, True, weight_seed=weightSeed)

        t0 = time.time()
        regressor.train("differential training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()

        print('differential predictions')
        print(predictions)

    return xAxis, yTest, dydxTest[:, deltidx], vegas, predvalues, preddeltas
    
    

@app.route('/model/price/instruments', methods=['POST'])
@cross_origin()
def getPredictedPriceForInstruments():
    if (request.method == 'POST'):
        instrumentList = request.get_json()
        response = getRawResponse()
        trainingTime = float(0)
        for item in instrumentList:
            instrumentId = item['instrumentId']
            spotPrice = item['spotprice']

            pricerModel = instrumentModelMap.get(instrumentId)
            print(pricerModel)
            if not pricerModel:
                print("No Model found for given instrument ", instrumentId)
                enrichResponse(response, instrumentId, 0, 0, 0, {})
                continue # "No model found for given Instrument"
            else:
                # Divide the spot prices by 100 to fit in the model
                spotPriceArr = (np.array(spotPrice)/100).reshape([-1, 1])
                predictions, deltas, prediction_time = predictPriceForInstrument(instrumentId, spotPriceArr,pricerModel)
                #predictions = predictions * 100
                trainingTime = 0 if not trainingTime else trainingTime
                totaltimetaken = trainingTime + prediction_time
                print(f'total time taken = {totaltimetaken}')
                original_spotPrice = np.array(spotPrice).reshape([-1,1])
                values = np.concatenate((original_spotPrice, predictions, deltas), axis=1)
                df = pd.DataFrame(values, columns=['spotPrice', 'predictedPrice', 'delta'])
                enrichResponse(response, instrumentId, prediction_time, trainingTime, totaltimetaken,
                               df.to_dict(orient="records"))

    return jsonify(response)



def initPriceModelforGivenInstrument(instrument,training_data):
    print('Initializing price Model')
    xTrain, yTrain, dydxTrain = getTrainingDataForGivenInstrument(instrument,training_data)
    model , trainingTime = prepareAndTrainModel(xTrain, yTrain, dydxTrain)
    return model , trainingTime


def getTrainingDataForGivenInstrument(instrument,training_data):
    df = pd.DataFrame(training_data)
    values_arr = np.array(df.values)
    xTrain = values_arr[:, :1].reshape([-1, 1])
    yTrain = values_arr[:, :2].reshape([-1, 1])
    dydxTrain = values_arr[:, :3].reshape([-1, 1])
    # print("training data", training_data)
    print(xTrain)
    print(yTrain)
    # print(dydxTrain)


    return xTrain, yTrain, dydxTrain

def prepareAndTrainModel(xTrain, yTrain, dydxTrain):
    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    
    print('Standard Model prep start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.prepare(size, False)
    print('Standard Model prep end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

    t0 = time.time()
    print('Standard Model Training start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.train("standard training")
    print('Standard Model Training end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))



    print('Model prep start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.prepare(size, True)
    print('Model prep end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

    
    print('Model Training start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.train("differential training")
    print('Model Training end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    t1 = time.time()

    training_time = t1 - t0
    print('Training time =', training_time)

    # instrumentModelMap.__setitem__(instrumentId, regressor)
    return regressor , training_time


def enrichResponse(rawResponse, instrumentId,prediction_time,trainingTime,totaltimetaken,values):
    rawResponse['data'].append({'instrumentId': instrumentId, 'predictionTime': prediction_time,
     'trainingTime': trainingTime, 'totalTime': totaltimetaken,
     'values': values})

def connectToRedis(db=0):

    REDIS_HOST = os.environ.get('REDIS_HOST')
    REDIS_PORT = os.environ.get('REDIS_PORT')
    redis_host = "13.68.144.231" if not REDIS_HOST else REDIS_HOST
    redis_port = 6379 if not REDIS_PORT else int(REDIS_PORT)
    print(f'Connecting to Redis host ={redis_host} , port ={redis_port}')
    rs = redis.StrictRedis(host=redis_host, port=redis_port, db=db, decode_responses=True)
    rs.ping()
    print("Connecting to Redis successfull")
    return rs


def fetchAllInstruments():
    instrumentList = []
    skippedInstrument = []

    try:
        redis_instrument_con = connectToRedis()
    except Exception as e:
        print('Could not connect to redis', str(e))
        error_message = 'Could not fetch instrument list from Redis '+ ' cause : ' + str(e)
        return instrumentList

    RANGE = os.environ.get('INSTRUMENT_RANGE')
    print(f'Instrument Range set for this app = {RANGE}')
    instrument_range = '1-5' if not RANGE else RANGE
    start_index = int(instrument_range.split('-')[0])-1
    end_index = int(instrument_range.split('-')[1])

    for key in redis_instrument_con.scan_iter():
        print(f'Instrument from redis = {key}')
        if (len(key) < 19):
            print(f'invalid instrument length ,  so skipping')
            skippedInstrument.append(key)
            continue
        instrumentList.append(key)

    print(f'Total Instrument found in cache ',len(instrumentList))
    print(f'Invalid Instruments = ', skippedInstrument)

    sublist = []
    if (end_index > len(instrumentList)):
        end_index = len(instrumentList)

    print(f'start Index = {start_index} , end index = {end_index}')
    try:
        sublist = instrumentList[start_index:end_index]
    except Exception as e :
        print('Could not fetch sublist ,some issue iwth instrument range')

    print('Processed Instrument as per Range = ',sublist)
    return sublist


def predictPriceForInstrument(instrumentId , spotPrice ,model):
    prediction_start_time = time.time()
    predictions, deltas = model.predict_values_and_derivs(spotPrice)
    prediction_time = time.time()-prediction_start_time
    return predictions , deltas , prediction_time

def generateTrainingDataAndTrainModel(instrumentId,seed,spotPrice,strikePrice,volatality,expiryInYears):
    generator = BlackScholes()
    simulSeed = seed
    generator.__init__(spot=(spotPrice), K=(strikePrice), vol=volatality, T2=(1 + expiryInYears))

    xTrain, yTrain, dydxTrain = generator.trainingSet(size, seed)

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print('Model prep start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    #regressor.prepare(size, True, weight_seed=simulSeed)
    #Seed only required for training set for model its None
    regressor.prepare(size, True, weight_seed=None)
    print('Model prep end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

    t0 = time.time()
    print('Model Training start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.train("differential training")
    print('Model Training end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    t1 = time.time()

    training_time = t1 - t0
    print('Training time =', training_time)

    #instrumentModelMap.__setitem__(instrumentId, regressor)
    return xTrain, yTrain, dydxTrain, regressor

def getRawResponse():
    return {"data":[]}

def getRequestParam(request_data):
    instrumentId = request_data['instrumentId']
    strikePrice = float(request_data['strikeprice'])
    expiryInYears = float(request_data['expiry'])
    spotPrice = float(request_data['spotprice'])
    volatality = float(request_data['volatility'])

    return instrumentId , strikePrice, expiryInYears, spotPrice, volatality

def populateModelCache(instrumentId, model):
    if not instrumentModelMap.get(instrumentId):
        print("Adding Model in cache for ", instrumentId)
        instrumentModelMap.__setitem__(instrumentId, model)
    else:
        print("Model Already Exist in cache")

def modelExist(instrumentId):
    return True if instrumentModelMap.get(instrumentId) else False



if __name__ == '__main__':
    print('REDIS HOST = ',os.environ.get('REDIS_HOST'))
    print('REDIS PORT = ',os.environ.get('REDIS_PORT'))
    print('INSTRUMENT RANGE = ',os.environ.get('INSTRUMENT_RANGE'))
    app.run(host="0.0.0.0", port=8090)
