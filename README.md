# Weather API

This repository houses the code to deploy the final fitted pytorch LSTM models as part of an API endpoint. To ensure that the models can actually be used outside of a local environment, I make use of FastAPI to deploy the model.

A flow diagram will be added to this

### API Architecture
As part of this project, I wanted to ensure that the model will always be readily available to be used, without needing to load the model on each call to make a prediction. Therefore, on API startup we query the MLFlow API to get the latest available model from the model registry. We then cache that model along with its associated pipeline to process any data when a predict request is being made. Additionally, we set up the model loading to check every hour if a new model has been added to the MLFlow model registry. It would then load this new model into cache, replacing the old model. This is so that we can hot-swap models on the fly without needing to take the application offline for a period of time. This will allow us to continually serve predictions. Also, since we have cached the model and pipline in memoery, we are able to get back lightning quick responses from the API for predictions, which greatly helps in systems where performance is key. 

### API Endpoints
There are 3 main endpoints that are part of this API: a simepl `get` request that returns a message if the API is live, a `get` method that is a healthcheck to determine if the model and processing pipeline are cached in the application, and a `post` request to actually predict on the supplied data.

Assoicated with the predict endpoint, we have a custom pydantic model that requires input to be a list of 16 time points. This is due to out LSTM model having a look-back window of 12 and are forecasting the next 4 hours of data points. So, whenever a request is made, we require the last 16 hours worth of data be supplied to the API. The model will then predict/forecast the air quality for the next hour, and return that back to the user in another pydantic model. I make use of pydantic as it has great data type validation, ensuring that we only pass data to the model that has the appropriate types. Now, in the event that the past covariates are not known in the future, they can simply be set to 0 and the model will handle this data

### Performance
As I said, since we cache the model at startup, we greatly reduce the amount of time taken to generate predictions. Rather than taling several seconds to load a model, we can transform the input and return the prediction from the model in around 1 second. This is great for systems where performance could be key. Not just for air quality, but any application where predictions and performance are critical. This system allows for extremely rapid response, ensuring that end-users do not need to wait long before they get their predictions.

### Example Request and Output
For the model, suppose we supply a sequence of time points like the following:

```json
{
  "sequence": [
    {
      "time": "2025-02-27T00:00:00",
      "pm10": 13.4,
      "pm2_5": 12.6,
      "carbon_monoxide": 352,
      "nitrogen_dioxide": 24.3,
      "sulphur_dioxide": 3.7,
      "ozone": 69,
      "uv_index": 1,
      "dust": 0.1,
      "aerosol_optical_depth": 0.1
    },
    {
      "time": "2025-02-27T01:00:00",
      "pm10": 13.3,
      "pm2_5": 12.6,
      "carbon_monoxide": 328,
      "nitrogen_dioxide": 21.2,
      "sulphur_dioxide": 3.2,
      "ozone": 72,
      "uv_index": 1,
      "dust": 0.09,
      "aerosol_optical_depth": 0.09
    },
    {
      "time": "2025-02-27T02:00:00",
      "pm10": 12.7,
      "pm2_5": 12,
      "carbon_monoxide": 276,
      "nitrogen_dioxide": 16.8,
      "sulphur_dioxide": 2.4,
      "ozone": 77,
      "uv_index": 1,
      "dust": 0.09,
      "aerosol_optical_depth": 0.09
    },
    {
      "time": "2025-02-27T03:00:00",
      "pm10": 11.1,
      "pm2_5": 10.5,
      "carbon_monoxide": 235,
      "nitrogen_dioxide": 12.9,
      "sulphur_dioxide": 1.8,
      "ozone": 81,
      "uv_index": 1,
      "dust": 0.09,
      "aerosol_optical_depth": 0.09
    },
    {
      "time": "2025-02-27T04:00:00",
      "pm10": 9.1,
      "pm2_5": 8.6,
      "carbon_monoxide": 224,
      "nitrogen_dioxide": 10.2,
      "sulphur_dioxide": 1.3,
      "ozone": 83,
      "uv_index": 1,
      "dust": 0.09,
      "aerosol_optical_depth": 0.09
    },
    {
      "time": "2025-02-27T05:00:00",
      "pm10": 7.7,
      "pm2_5": 7.4,
      "carbon_monoxide": 225,
      "nitrogen_dioxide": 8,
      "sulphur_dioxide": 1,
      "ozone": 84,
      "uv_index": 0,
      "dust": 0.09,
      "aerosol_optical_depth": 0.09
    },
    {
      "time": "2025-02-27T06:00:00",
      "pm10": 7,
      "pm2_5": 6.8,
      "carbon_monoxide": 224,
      "nitrogen_dioxide": 6.4,
      "sulphur_dioxide": 0.7,
      "ozone": 84,
      "uv_index": 0,
      "dust": 0.1,
      "aerosol_optical_depth": 0.1
    },
    {
      "time": "2025-02-27T07:00:00",
      "pm10": 6.5,
      "pm2_5": 6.4,
      "carbon_monoxide": 217,
      "nitrogen_dioxide": 5.4,
      "sulphur_dioxide": 0.6,
      "ozone": 84,
      "uv_index": 0,
      "dust": 0.12,
      "aerosol_optical_depth": 0.12
    },
    {
      "time": "2025-02-27T08:00:00",
      "pm10": 6.2,
      "pm2_5": 6.1,
      "carbon_monoxide": 208,
      "nitrogen_dioxide": 5.1,
      "sulphur_dioxide": 0.7,
      "ozone": 84,
      "uv_index": 0,
      "dust": 0.13,
      "aerosol_optical_depth": 0.13
    },
    {
      "time": "2025-02-27T09:00:00",
      "pm10": 6.1,
      "pm2_5": 6,
      "carbon_monoxide": 202,
      "nitrogen_dioxide": 5.1,
      "sulphur_dioxide": 0.7,
      "ozone": 83,
      "uv_index": 0,
      "dust": 0.16,
      "aerosol_optical_depth": 0.16
    },
    {
      "time": "2025-02-27T10:00:00",
      "pm10": 5.8,
      "pm2_5": 5.8,
      "carbon_monoxide": 200,
      "nitrogen_dioxide": 5.6,
      "sulphur_dioxide": 0.7,
      "ozone": 82,
      "uv_index": 0,
      "dust": 0.2,
      "aerosol_optical_depth": 0.2
    },
    {
      "time": "2025-02-27T11:00:00",
      "pm10": 5.1,
      "pm2_5": 5,
      "carbon_monoxide": 200,
      "nitrogen_dioxide": 6.5,
      "sulphur_dioxide": 0.7,
      "ozone": 81,
      "uv_index": 0,
      "dust": 0.19,
      "aerosol_optical_depth": 0.19
    },
    {
      "time": "2025-02-27T12:00:00",
      "pm10": 0,
      "pm2_5": 0,
      "carbon_monoxide": 0,
      "nitrogen_dioxide": 0,
      "sulphur_dioxide": 0,
      "ozone": 0,
      "uv_index": 0,
      "dust": 0,
      "aerosol_optical_depth": 0
    },
    {
      "time": "2025-02-27T13:00:00",
      "pm10": 0,
      "pm2_5": 0,
      "carbon_monoxide": 0,
      "nitrogen_dioxide": 0,
      "sulphur_dioxide": 0,
      "ozone": 0,
      "uv_index": 0,
      "dust": 0,
      "aerosol_optical_depth": 0
    },
    {
      "time": "2025-02-27T14:00:00",
      "pm10": 0,
      "pm2_5": 0,
      "carbon_monoxide": 0,
      "nitrogen_dioxide": 0,
      "sulphur_dioxide": 0,
      "ozone": 0,
      "uv_index": 0,
      "dust": 0,
      "aerosol_optical_depth": 0
    },
    {
      "time": "2025-02-27T15:00:00",
      "pm10": 0,
      "pm2_5": 0,
      "carbon_monoxide": 0,
      "nitrogen_dioxide": 0,
      "sulphur_dioxide": 0,
      "ozone": 0,
      "uv_index": 0,
      "dust": 0,
      "aerosol_optical_depth": 0
    }
  ]
}
```

When this is submitted to the API, we should expect a response of the following:
```json
{
  "prediction": [
    [
      41.69682693481445,
      42.849639892578125,
      41.96396255493164,
      42.51116943359375
    ]
  ]
}
```

This will be our forecasted air quality for the next 4 hours from the last know past covariates.

If we were to send a request with `curl` it would look like the following:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sequence": [
    ...
  ]
}'
```