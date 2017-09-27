# CAMP-ANN-training

## Summary
Prototype of neural network training service and prediction API

## Requirements

For training EC2 with GPU accelator, tensorflow,cuda kit,pandas are required. 

## Raw training data

Raw training data should be CSV file located in the 'csvdata' directory in the same loation as the programm. 

## Programm list
### 
* emb.py :


## Install

    $ git clone ssh://git-codecommit.eu-west-1.amazonaws.com/v1/repos/camp-frontend
    $ cd camp-frontend
    $ yarn install

### Configure app

To be able to run the application a .env file is required and need to be located in "/camp-frontend/.env". This file holds information such as backend-url, aws-userpool-id etc. This file should not be checked into our GIT repository and is therefor added to our .gitignore file. There is a env.sample file that has all the proper keys but without values that should be used for a newcommer to the project. Rename env.sample to .env and ask a developer in the project about the values for the environment keys.

## Start & watch

	$ npm start
	
## Run Unit tests & watch
	$ npm run test

## Run UI-Browser tests
	// Start server
	$ npm run e2e-server
	// Then run tests
	$ npm run e2e

## Build for production

    $ npm run build

## Adding new npm-packages
	// To add a runtime dependency
	$ yarn add redux
	// To add a test/development dependency
	$ yarn add jest --dev
	// Check in /yarn.lock  /package.json

