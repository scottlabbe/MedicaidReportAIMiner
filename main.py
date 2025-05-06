import logging
from app import app
from routes import register_routes

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Register routes
register_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
