Inference should take place on a remote GPU from the Modal app. 
- A `modal.Volume` caches the downloaded HF weights so repeated runs don't re-download the model each time.
- Inference runs should be resumable so we don't waste compute after losing Internet connection.
- Models should be passed as parameters to training runs so that we can run the same prompt battery on different models.