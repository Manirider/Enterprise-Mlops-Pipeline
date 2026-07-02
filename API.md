# API Specifications

## FastAPI Endpoints

### `/health`
- **Method:** GET
- **Description:** Verifies that the API service is active and the registered model artifact is loaded.

### `/predict`
- **Method:** POST
- **Request Format:**
  ```json
  {
    "features": [0.1, 0.2, 0.3, 0.4]
  }
  ```
- **Response Format:**
  ```json
  {
    "prediction": 1,
    "confidence": 0.94
  }
  ```

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)