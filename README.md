# Weather API

This repository houses the code to deploy the final fitted pytorch LSTM models as part of an API endpoint. To ensure that the models can actually be used outside of a local environment, I make use of FastAPI to deploy the model.

A flow diagram will be added to this

### API Architecture
As part of this project, I wanted to ensure that the model will always be readily available to be used, without needing to load the model on each call to make a prediction. Therefore, on API startup we query the MLFlow API to get the latest available model from the model registry. We then cache that model along with its associated pipeline to process any data when a predict request is being made. Additionally, we set up the model loading to check every hour if a new model has been added to the MLFlow model registry. It would then load this new model into cache, replacing the old model. This is so that we can hot-swap models on the fly without needing to take the application offline for a period of time. This will allow us to continually serve predictions. Also, since we have cached the model and pipline in memoery, we are able to get back lightning quick responses from the API for predictions, which greatly helps in systems where performance is key. 

### API Endpoints
There are 3 main endpoints that are part of this API: a simepl `get` request that returns a message if the API is live, a `get` method that is a healthcheck to determine if the model and processing pipeline are cached in the application, and a `post` request to actually predict on the supplied data.

Assoicated with the predict endpoint, we have a custom pydantic model that requires input to be a list of 12 time points. This is due to out LSTM model having a look-back window of 12. So, whenever a request is made, we require the last 12 hours worth of data be supplied to the API. The model will then predict/forecast the air quality for the next hour, and return that back to the user in another pydantic model. I make use of pydantic as it has great data type validation, ensuring that we only pass data to the model that has the appropriate types. 

### Performance
As I said, since we cache the model at startup, we greatly reduce the amount of time taken to generate predictions. Rather than taling several seconds to load a model, we can transform the input and return the prediction from the model in around 1 second. This is great for systems where performance could be key. Not just for air quality, but any application where predictions and performance are critical. This system allows for extremely rapid response, ensuring that end-users do not need to wait long before they get their predictions.