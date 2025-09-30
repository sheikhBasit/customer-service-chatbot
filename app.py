from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import AsyncIterator, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from middlewares.error_handler import ErrorHandlingMiddleware
from middlewares.limit_handler import LimitRequestSizeMiddleware
from middlewares.logging_handler import LoggingMiddleware
from middlewares.security_handler import SecurityHeadersMiddleware
from services.diagnostic_agent import create_diagnostic_agent
from services.multimodal_embeddings import initialize_clip_model
from services.image_analyzer import ImageAnalyzer
# from services.vectorstore import process_pdf_with_images
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from services.intent_classifier import IntentClassifier
from services.simple_responses import SimpleResponseGenerator
# App imports
from routes import chat
from config import settings
from utils.logging import configure_logging
from database import connect_to_mongo, close_mongo_connection
import asyncio

# Initialize logging
logger = logging.getLogger(__name__)
configure_logging()

# Global service instances
diagnostic_agent = None
image_analyzer = None
vectorstore = None
image_data_store = None

async def initialize_services():
    """Initialize all services with retries and error handling"""
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing services (attempt {attempt + 1}/{max_retries})")
            

             # Initialize intent classifier
            logger.info("Initializing intent classifier...")
            app.state.intent_classifier = IntentClassifier(settings.GROQ_API_KEY)
            logger.info("Intent classifier initialized successfully")

            # Initialize diagnostic agent
            logger.info("Initializing diagnostic agent...")
            app.state.diagnostic_agent = create_diagnostic_agent(settings.GROQ_API_KEY)
            logger.info("Diagnostic agent initialized successfully")

            # Initialize image analyzer
            logger.info("Initializing image analyzer...")
            analyzer = ImageAnalyzer(hf_token=settings.HF_TOKEN)
            await analyzer.initialize()
            app.state.image_analyzer = analyzer
            logger.info("Image analyzer initialized successfully")

            # Load FAISS index and image data from cache
            logger.info("Loading FAISS index and image data from cache...")
            from services.vector_cache import VectorCache

            cache = VectorCache(settings.VECTOR_CACHE_DIR)
            cache_key = cache.get_cache_key(settings.KNOWLEDGE_BASE_PDF)
            
            # Check if cache exists, if not create it
            if cache.cache_exists(cache_key):
                app.state.vectorstore, app.state.image_data_store = cache.load_from_cache(cache_key)
                logger.info("Vectorstore loaded successfully from cache.")
            else:
                logger.info("Cache not found, processing PDF...")
                from services.vectorstore import process_pdf_with_images
                app.state.vectorstore, app.state.image_data_store = process_pdf_with_images(
                    settings.KNOWLEDGE_BASE_PDF,
                    cache_dir=settings.VECTOR_CACHE_DIR
                )
                logger.info("Vectorstore created successfully")

            logger.info("All services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Service initialization failed (attempt {attempt + 1}): {e}", exc_info=True)
            if attempt == max_retries - 1:
                logger.critical("Max retries reached. Continuing without some services.")
                return False
            await asyncio.sleep(retry_delay)
            

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events"""
    try:
        # -------------------
        # Startup logic
        # -------------------
        logger.info("Connecting to MongoDB...")
        app.state.mongo_client = await connect_to_mongo()  # Save client to app.state
        logger.info("MongoDB connection established and saved to app.state.mongo_client")

        logger.info("Initializing application services...")
        if not await initialize_services():
            raise RuntimeError("Failed to initialize services after multiple attempts")

        # Eagerly load the CLIP model
        logger.info("Initializing CLIP model on startup...")
        initialize_clip_model()
        logger.info("CLIP model loaded successfully.")

        # Setup MongoDB geospatial index and migrate existing data
        from services.mechanics import MechanicService
        logger.info("Setting up MongoDB geospatial index...")
        await MechanicService.create_geospatial_index()
        await MechanicService.migrate_existing_to_geospatial()
        logger.info("MongoDB geospatial setup completed successfully")

        # Yield control to FastAPI
        yield

    except Exception as e:
        logger.critical(f"Application startup failed: {str(e)}", exc_info=True)
        raise

    finally:
        # -------------------
        # Shutdown logic
        # -------------------
        logger.info("Shutting down services...")
        if hasattr(app.state, "mongo_client") and app.state.mongo_client:
            await close_mongo_connection()
            logger.info("MongoDB connection closed")
        logger.info("Services shutdown complete")
        
# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    openapi_version="3.0.3"
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LimitRequestSizeMiddleware, max_content_length=1024 * 1024)  # 1MB
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(chat.router)


@app.get("/")
async def get_status(request: Request):
    """Health check endpoint"""
    services_status = {
        "diagnostic_agent": "ready" if diagnostic_agent else "not initialized",
        "image_analyzer": "ready" if image_analyzer else "not initialized",
        "vectorstore": "ready" if vectorstore else "not initialized",
        "mongodb": "connected"
    }
    return {
        "status": "ok", 
        "service": settings.APP_NAME,
        "services": services_status,
        "version": "1.0.0"
    }
# @app.get("/api/docs", include_in_schema=False)
# async def custom_swagger_ui_html():
#     return get_swagger_ui_html(
#         openapi_url=app.openapi_url,
#         title=app.title + " - Swagger UI",
#         swagger_js_url="/static/swagger-ui-bundle.js",
#         swagger_css_url="/static/swagger-ui.css",
#         swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
#     )

# def get_diagnostic_agent():
#     """Get the initialized diagnostic agent"""
#     if not diagnostic_agent:
#         raise RuntimeError("Diagnostic agent not initialized")
#     return diagnostic_agent

# def get_image_analyzer():
#     """Get the initialized image analyzer"""
#     if not image_analyzer:
#         raise RuntimeError("Image analyzer not initialized")
#     return image_analyzer

# def get_vectorstore():
#     """Get the initialized vectorstore and image data store"""
#     if not vectorstore:
#         raise RuntimeError("Vectorstore not initialized")
#     return vectorstore, image_data_store