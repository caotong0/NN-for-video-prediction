import engine
import utility

if __name__ == '__main__':

    args = utility.loadParams(jsonFile = 'config.json')
    logger = utility.Tee('./log.txt', 'a')

    print('StarBriNet is running!')
    print('-------------------------------------------------')
    print('Model: ' + args['model']['arch'])
    print('Optimizer: ' + args['model']['optimizer'])
    print('Batch Size: ' + str(args['data']['train']['batchSize']))
    print('-------------------------------------------------')
    
    runner = engine.Engine(args)

    numEpochs = args['model']['epochs']
    testFreq = args['model']['testFreq']
    if args['Training']:
        for e in range(runner.epoch, numEpochs):
            runner.train()
            if e % testFreq == 0:
                runner.validate(e,True)
    else:
        for e in range(1, 2):
            runner.validate(e, False)
