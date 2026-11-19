# Architecture

The React client communicates with FastAPI over REST. The API validates inputs, delegates inference, and exposes optional history and generated artifacts. Model code is isolated under `ml/src` and can be used by training scripts or the backend adapter. MongoDB is an optional persistence layer; analysis remains available without it.
